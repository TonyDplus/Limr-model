# -*- conding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, SubsetRandomSampler

from tools.dataset_class import *
from tools.metric import metric
from tools.utils import *

curPath = os.path.abspath(os.path.dirname('__file__'))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import nltk
import os

class MTFMPPConfig(object):
    def __init__(self, ds_config):
        with open(rootPath + '/data/api_quality_feature.dat', 'r') as f:
            self.api_quality = [line.split('::') for line in f]
        del (self.api_quality[0])
        for api in self.api_quality:
            api[0] = api[0].split('/')[-1].replace(' ', '-').lower()
            api[1] = api[1].split(',')
            for i, ele in enumerate(api[1]):
                try:
                    api[1][i] = eval(ele)
                except:
                    api[1][i] = 1.0
            api[1] = torch.Tensor(api[1])
        api_list = ds_config.api_ds.name
        self.api_quality_embed = torch.zeros(len(api_list), 13)
        for api_tmp in self.api_quality:
            try:
                self.api_quality_embed[api_list.index(api_tmp[0])] = api_tmp[1]
            except:
                pass

        self.api_tag_embed = torch.zeros(len(ds_config.api_ds), ds_config.api_ds.num_category)
        for i, api in enumerate(ds_config.api_ds):
            self.api_tag_embed[i] = api[2]

        self.model_name = 'MTFM++_simple_opt'
        self.embed_dim = ds_config.embed_dim
        self.max_doc_len = ds_config.max_doc_len
        self.num_category = ds_config.num_category
        self.num_category_en = ds_config.num_category_en
        self.feature_dim = 36
        self.num_kernel = 128
        self.dropout = 0.1  # 优化1: 降低Dropout率 (0.2 -> 0.1)
        self.kernel_size = [2, 3, 4, 5]
        self.num_mashup = ds_config.num_mashup
        self.num_api = ds_config.num_api
        self.vocab_size = ds_config.vocab_size
        self.embed = ds_config.embed
        self.lr = 5e-4  # 优化2: 降低学习率 (1e-3 -> 5e-4)
        self.batch_size = 128
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')


class MTFMPP(nn.Module):
    def __init__(self, config):
        super(MTFMPP, self).__init__()
        if config.embed is not None:
            self.embed_layer = nn.Embedding.from_pretrained(config.embed, freeze=False)
        else:
            self.embed_layer = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=config.vocab_size - 1)

        self.api_quality_embed = nn.Embedding.from_pretrained(config.api_quality_embed, freeze=True)
        self.api_quality_layer = nn.Linear(in_features=13, out_features=1)

        self.api_tag_embed = nn.Embedding.from_pretrained(config.api_tag_embed, freeze=True)
        self.api_tag_layer = nn.Linear(in_features=config.num_category, out_features=config.feature_dim)

        self.api_sc_convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=config.embed_dim,
                                    out_channels=config.num_kernel,
                                    kernel_size=h),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=config.max_doc_len - h + 1))
            for h in config.kernel_size
        ])
        self.api_sc_output = nn.Linear(in_features=config.num_kernel * len(config.kernel_size),
                                       out_features=config.feature_dim)

        self.api_fusion_layer = nn.Linear(in_features=config.feature_dim * 2, out_features=config.feature_dim)


        self.category_en_embed = nn.Embedding(config.num_category_en, 24)  # 增加嵌入维度
        self.category_en_fc = nn.Sequential(
            nn.Linear(24, config.feature_dim),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(config.feature_dim, config.feature_dim // 2)
        )
        

        self.semantic_transform = nn.Sequential(
            nn.Linear(config.num_api, config.num_api),
            nn.BatchNorm1d(config.num_api),  # 使用BatchNorm替代LayerNorm
            nn.ReLU(),
            nn.Dropout(0.05)  # 降低Transform层的Dropout
        )
        
        self.interaction_transform = nn.Sequential(
            nn.Linear(config.num_api, config.num_api),
            nn.BatchNorm1d(config.num_api),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        
        self.post_fusion_transform = nn.Sequential(
            nn.Linear(config.num_api, config.num_api),
            nn.BatchNorm1d(config.num_api),
            nn.ReLU(),
            nn.Dropout(0.05)
        )
        

        self.api_pre_transform = nn.Sequential(
            nn.Linear(config.num_api, config.num_api),
            nn.Tanh(),
            nn.Dropout(0.05)
        )

        self.sc_convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=config.embed_dim,
                                    out_channels=config.num_kernel,
                                    kernel_size=h),
                          nn.ReLU(),
                          nn.MaxPool1d(kernel_size=config.max_doc_len - h + 1))
            for h in config.kernel_size
        ])
        self.sc_output = nn.Linear(in_features=config.num_kernel * len(config.kernel_size), out_features=config.num_api)

        self.fic_input = nn.Linear(in_features=config.num_kernel * len(config.kernel_size),
                                   out_features=config.feature_dim)
        self.fic_fcl = nn.Linear(config.num_api * 2, config.num_api)

        self.fusion_layer = nn.Linear(config.num_api * 3 + config.feature_dim // 2, config.num_api)


        from tools.KAN import FastKANLayer
        
        self.api_task_layer = FastKANLayer(
            input_dim=config.num_api,
            output_dim=config.num_api,
            grid_min=-0.4,
            grid_max=0.4,
            num_grids=4,
            use_base_update=True,
            base_activation=F.tanh,
            spline_weight_init_scale=0.004
        )
        
        self.category_task_layer = nn.Linear(config.num_api, config.num_category)

        self.dropout = nn.Dropout(config.dropout)
        self.logistic = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        

        self.residual_weight = nn.Parameter(torch.tensor(0.15))
        
        self._init_weights()

    def _init_weights(self):

        for module in [self.semantic_transform, self.interaction_transform, self.post_fusion_transform]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0.0)

    def forward(self, mashup_des, api_des, category_en_idx=None):
        # api semantic component
        api_embed = self.embed_layer(api_des)
        api_embed = api_embed.permute(0, 2, 1)
        e = [conv(api_embed) for conv in self.api_sc_convs]
        e = torch.cat(e, dim=2)
        e = e.view(e.size(0), -1)
        api_sc = self.api_sc_output(e)
        api_sc = self.tanh(api_sc)
        api_sc = api_sc.permute(1, 0)

        # api tag layer
        api_tag_value = self.api_tag_layer(self.api_tag_embed.weight)
        api_tag_value = api_tag_value.permute(1, 0)
        api_tag_value = self.tanh(api_tag_value)

        # semantic component
        embed = self.embed_layer(mashup_des)
        embed = embed.permute(0, 2, 1)
        e = [conv(embed) for conv in self.sc_convs]
        e = torch.cat(e, dim=2)
        e = e.view(e.size(0), -1)
        u_sc = self.sc_output(e)

        # feature interaction component
        u_sc_trans = self.fic_input(e)
        u_sc_trans = self.tanh(u_sc_trans)
        u_mm = torch.matmul(u_sc_trans, self.tanh(api_sc + api_tag_value))
        u_fic = self.tanh(u_mm)

        # api quality layer
        api_quality_value = self.api_quality_layer(self.api_quality_embed.weight).permute(1, 0)
        api_quality_value = self.tanh(api_quality_value)


        if category_en_idx is not None:
            category_en_embed = self.category_en_embed(category_en_idx)
            category_en_features = self.tanh(self.category_en_fc(category_en_embed))
        else:
            category_en_features = torch.zeros(u_sc.size(0), self.category_en_fc[-1].out_features).to(u_sc.device)


        u_sc_transformed = self.semantic_transform(u_sc)
        u_sc = u_sc + self.residual_weight * u_sc_transformed

        u_fic_transformed = self.interaction_transform(u_fic)
        u_fic = u_fic + self.residual_weight * u_fic_transformed

        api_quality_expanded = api_quality_value.expand_as(u_sc)

        # fusion layer
        u_mmf = self.fusion_layer(torch.cat((u_sc, u_fic, api_quality_expanded, category_en_features), dim=1))

        # 融合后变换
        u_mmf_transformed = self.post_fusion_transform(u_mmf)
        u_mmf = u_mmf + self.residual_weight * u_mmf_transformed

        # dropout
        u_mmf = self.dropout(u_mmf)

        # 任务专用预处理
        api_features = self.api_pre_transform(u_mmf)
        api_features = api_features + self.residual_weight * u_mmf

        y_m = self.api_task_layer(api_features)
        z_m = self.category_task_layer(u_mmf)

        return self.logistic(y_m), self.logistic(z_m)


class Train(object):
    def __init__(self, input_model, input_config, train_iter, test_iter, val_iter, case_iter, log, input_ds,
                 model_path=None):
        self.model = input_model
        self.config = input_config
        self.train_iter = train_iter
        self.test_iter = test_iter
        self.val_iter = val_iter
        self.case_iter = case_iter
        self.api_cri = torch.nn.BCELoss()

        self.optim = torch.optim.AdamW(input_model.parameters(), lr=self.config.lr, weight_decay=5e-5)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optim, mode='max', factor=0.7, patience=5, verbose=True, min_lr=1e-5
        )
        
        self.epoch = 100
        self.top_k_list = [1, 5, 10, 15, 20, 25, 30]
        self.log = log
        self.ds = input_ds
        if model_path:
            self.model_path = model_path
        else:
            self.model_path = 'checkpoint/%s.pth' % self.config.model_name

        self.early_stopping = EarlyStopping(patience=12, path=self.model_path, monitor_metric='ndcg')
        self.api_des = torch.LongTensor(self.ds.api_ds.description).to(self.config.device)

    def train(self):
        data_iter = self.train_iter
        print('开始训练...')

        for epoch in range(self.epoch):
            self.model.train()
            api_loss = []
            category_loss = []

            for batch_idx, batch_data in enumerate(data_iter):
                index = batch_data[0].to(self.config.device)
                des = batch_data[1].to(self.config.device)
                category_target = batch_data[2].float().to(self.config.device)
                api_target = batch_data[3].float().to(self.config.device)
                category_en_idx = batch_data[6].to(self.config.device)

                self.optim.zero_grad()
                api_pred, category_pred = self.model(des, self.api_des, category_en_idx)

                api_loss_ = self.api_cri(api_pred, api_target)
                category_loss_ = self.cate_cri(category_pred, category_target)

                epoch_progress = epoch / self.epoch
                api_weight = 1.0 + 0.3 * epoch_progress
                category_weight = 1.0 - 0.2 * epoch_progress
                
                loss_ = api_weight * api_loss_ + category_weight * category_loss_
                
                loss_.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                
                self.optim.step()
                api_loss.append(api_loss_.item())
                category_loss.append(category_loss_.item())

            api_loss = np.average(api_loss)
            category_loss = np.average(category_loss)

            info = '[Epoch:%d] ApiLoss:%.6f CateLoss:%.6f' % (epoch + 1, api_loss, category_loss)
            print(info)
            self.log.write(info + '\n')
            self.log.flush()
            
            val_ndcg = self.evaluate(test=False)
            self.scheduler.step(val_ndcg)
            self.early_stopping(float(val_ndcg), self.model, 'NDCG@5')

            if self.early_stopping.early_stop:
                print("Early stopping")
                break

    def evaluate(self, test=None):
        if test:
            data_iter = self.test_iter
            label = 'Test'
            print('开始测试...')
        else:
            data_iter = self.val_iter
            label = 'Evaluate'
        self.model.eval()

        ndcg_a = np.zeros(len(self.top_k_list))
        recall_a = np.zeros(len(self.top_k_list))
        ap_a = np.zeros(len(self.top_k_list))
        pre_a = np.zeros(len(self.top_k_list))
        ndcg_c = np.zeros(len(self.top_k_list))
        recall_c = np.zeros(len(self.top_k_list))
        ap_c = np.zeros(len(self.top_k_list))
        pre_c = np.zeros(len(self.top_k_list))

        api_loss = []
        category_loss = []
        num_batch = len(data_iter)

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_iter):
                index = batch_data[0].to(self.config.device)
                des = batch_data[1].to(self.config.device)
                category_target = batch_data[2].float().to(self.config.device)
                api_target = batch_data[3].float().to(self.config.device)
                category_en_idx = batch_data[6].to(self.config.device)
                
                api_pred, category_pred = self.model(des, self.api_des, category_en_idx)
                
                api_loss_ = self.api_cri(api_pred, api_target)
                category_loss_ = self.cate_cri(category_pred, category_target)
                api_loss.append(api_loss_.item())
                category_loss.append(category_loss_.item())

                api_pred = api_pred.cpu().detach()
                category_pred = category_pred.cpu().detach()

                ndcg_, recall_, ap_, pre_ = metric(batch_data[3], api_pred, top_k_list=self.top_k_list)
                ndcg_a += ndcg_
                recall_a += recall_
                ap_a += ap_
                pre_a += pre_

                ndcg_, recall_, ap_, pre_ = metric(batch_data[2], category_pred, top_k_list=self.top_k_list)
                ndcg_c += ndcg_
                recall_c += recall_
                ap_c += ap_
                pre_c += pre_

        api_loss = np.average(api_loss)
        category_loss = np.average(category_loss)

        ndcg_a = ndcg_a / num_batch
        recall_a = recall_a / num_batch
        ap_a = ap_a / num_batch
        pre_a = pre_a / num_batch

        ndcg_c = ndcg_c / num_batch
        recall_c = recall_c / num_batch
        ap_c = ap_c / num_batch
        pre_c = pre_c / num_batch

        if test:
            info = '[%s] ApiLoss: %.6f, CateLoss: %.6f' % (label, api_loss, category_loss)
            print(info)
            self.log.write(info + '\n')
            
            print('=== API推荐性能 ===')
            self.log.write('=== API推荐性能 ===\n')
            
            display_indices = [0, 1, 2, 3, 6]
            display_k = [self.top_k_list[i] for i in display_indices]
            
            ndcg_str = 'NDCG@' + ', NDCG@'.join([f'{k}: {ndcg_a[i]:.4f}' for i, k in zip(display_indices, display_k)])
            map_str = 'MAP@' + ', MAP@'.join([f'{k}: {ap_a[i]:.4f}' for i, k in zip(display_indices, display_k)])
            pre_str = 'Pre@' + ', Pre@'.join([f'{k}: {pre_a[i]:.4f}' for i, k in zip(display_indices, display_k)])
            rec_str = 'Rec@' + ', Rec@'.join([f'{k}: {recall_a[i]:.4f}' for i, k in zip(display_indices, display_k)])
            
            print(ndcg_str)
            print(map_str)
            print(pre_str)
            print(rec_str)
            
            self.log.write(ndcg_str + '\n')
            self.log.write(map_str + '\n')
            self.log.write(pre_str + '\n')
            self.log.write(rec_str + '\n')
            self.log.flush()
        else:
            # 验证阶段也显示关键指标
            print(f'  Val NDCG@5: {ndcg_a[1]:.4f}, NDCG@10: {ndcg_a[2]:.4f}, MAP@5: {ap_a[1]:.4f}')

        return ndcg_a[1]

    def case_analysis(self):
        data_iter = self.case_iter
        self.model.eval()

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(data_iter):
                index = batch_data[0].to(self.config.device)
                des = batch_data[1].to(self.config.device)
                category_en_idx = batch_data[6].to(self.config.device)
                
                api_pred, category_pred = self.model(des, self.api_des, category_en_idx)
                
                api_pred = api_pred.cpu().detach()
                category_pred = category_pred.cpu().detach()

                for i in range(len(index)):
                    self.log.write("Mashup_index:" + str(index[i].cpu().item()) + '\n')
                    self.log.write("Mashup_des:" + str([self.ds.mashup_ds.vocab[word] for word in des[i].cpu() if word != len(self.ds.mashup_ds.vocab)]) + '\n')
                    
                    _, y_m_index = torch.topk(api_pred[i], 15)
                    self.log.write("Recommended API: " + str([self.ds.api_ds.name[index] for index in y_m_index]) + '\n')
                    
                    api_ground_truth = [self.ds.api_ds.name[j] for j in range(len(batch_data[3][i])) if batch_data[3][i][j] == 1]
                    self.log.write("API ground truth: " + str(api_ground_truth) + '\n')
                    self.log.write('\n')
                    
        self.log.flush()


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_name', type=str, default='LIMR')
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(2020)
    np.random.seed(2020)
    torch.manual_seed(2020)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(2020)
        torch.backends.cudnn.deterministic = True
    
    print("正在加载数据集...")
    ds = TextDataset()
    
    config = MTFMPPConfig(ds)
    model = MTFMPP(config).to(config.device)
    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # 数据分割和加载器创建
    from tools.utils import get_indices
    train_indices, val_indices, test_indices = get_indices(ds.mashup_ds)
    
    train_iter = DataLoader(ds.mashup_ds, batch_size=config.batch_size, sampler=SubsetRandomSampler(train_indices))
    val_iter = DataLoader(ds.mashup_ds, batch_size=config.batch_size, sampler=SubsetRandomSampler(val_indices))
    test_iter = DataLoader(ds.mashup_ds, batch_size=config.batch_size, sampler=SubsetRandomSampler(test_indices))
    case_iter = DataLoader(ds.mashup_ds, batch_size=config.batch_size, sampler=SubsetRandomSampler(test_indices[:100]))
    
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, f'{args.log_name}_{time.strftime("%Y%m%d_%H%M%S")}.log')
    log = open(log_file, 'w')
    
    trainer = Train(model, config, train_iter, test_iter, val_iter, case_iter, log, ds)
    trainer.train()
    
    model.load_state_dict(torch.load(trainer.model_path))
    trainer.evaluate(test=True)
    
    log.close()
    print(f"训练完成，日志保存到: {log_file}")


if __name__ == '__main__':
    main() 