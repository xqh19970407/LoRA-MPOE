import torch 
import numpy as np
from torch.nn import GELU
from torch import nn
from inspect import isfunction
import math
# from mpo_lab.Matrix2MPO import MPO
class Config():
    def __init__(self):
        self.linear_trunc = 10000000
        self.n_embd = 1024
        self.train_AT_index = [2]
def default(val, default_val):
    default_val = default_val() if isfunction(default_val) else default_val
    return val if val is not None else default_val
def init_(t):
    dim = t.shape[-1]
    std = 1 / math.sqrt(dim)
    return t.uniform_(-std, std)
def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)
config = Config()

class MPO:
    def __init__(self, mpo_input_shape, mpo_output_shape, truncate_num, fix_rank=None):
        self.mpo_input_shape = mpo_input_shape
        self.mpo_output_shape = mpo_output_shape
        self.truncate_num = truncate_num
        self.num_dim = len(mpo_input_shape)
        self.mpo_ranks = self.compute_rank(truncate_num=None)
        if fix_rank:
            self.mpo_truncate_ranks = fix_rank
        else:
            self.mpo_truncate_ranks = self.compute_rank(truncate_num=self.truncate_num)

    def compute_rank_position(self, s, truncate_num=None):

        """
        Calculate the rank position in MPO bond dimension
        :param s: target bond ,type = int, range in [1:len(mpo_input_shape-1)], r_0 = r_n = 1.
        :return:  target bond 's' real bond dimension.
        """
        rank_left = 1  # ranks_left: all the shape multiply in left of 's'.
        rank_right = 1  # ranks_right: all the shape multiply in right of 's'.
        for i in range(0, s):
            rank_left = rank_left * self.mpo_input_shape[i] * self.mpo_output_shape[i]
        for i in range(s, self.num_dim):
            rank_right = rank_right * self.mpo_input_shape[i] * self.mpo_output_shape[i]
        if truncate_num == None:
            min_rank = min(rank_left, rank_right)
        else:
            min_rank = min(int(self.truncate_num), rank_left, rank_right)
        return min_rank

    def compute_rank(self, truncate_num):
        """
        :param mpo_input_shape: the input mpo shape, type = list. [i0,i1,i2,...,i_(n-1)]
        :param truncate_num: the truncate number of mpo, type = int.
        :return:max bond dimension in every bond position, type = list, [r0,r1,r2,...,r_n],r0=r_n=1
        """
        bond_dims = [1 for i in range(self.num_dim + 1)]
        for i in range(1, self.num_dim):
            bond_dims[i] = self.compute_rank_position(i, truncate_num)
        return bond_dims

    def get_tensor_set(self, inp_matrix):
        """
        Calculate the left canonical of input matrix with a given mpo_input_shape
        :param inp_matrix: the input matrix
        :param mpo_input_shape:
        :return: a tensor with left canonical in input matrix
        """
        tensor_set = []
        res = inp_matrix
        #################################################################################
        
        res = res.reshape(tuple(self.mpo_input_shape[:]) + tuple(self.mpo_output_shape[:]))
        self.index_permute = np.transpose(
            np.array(range(len(self.mpo_input_shape) + len(self.mpo_output_shape))).reshape((2, -1))).flatten()
        res = np.transpose(res, self.index_permute)
        #################################################################################
        for i in range(self.num_dim - 1):
            # Do the SVD operator
            res = res.reshape([self.mpo_ranks[i] * self.mpo_input_shape[i] * self.mpo_output_shape[i], -1])
            u, lamda, v = np.linalg.svd(res, full_matrices=False)
            # The first tensor should be T1(r_i+1, m_i, n_i, r_i)
            u = u.reshape([self.mpo_ranks[i], self.mpo_input_shape[i], self.mpo_output_shape[i], self.mpo_ranks[i+1]])
            tensor_set.append(u)
            res = np.dot(np.diag(lamda), v)
        res = res.reshape([self.mpo_ranks[self.num_dim-1], self.mpo_input_shape[self.num_dim-1],
                           self.mpo_output_shape[self.num_dim-1], self.mpo_ranks[self.num_dim]])
        tensor_set.append(res)
        return tensor_set
    def left_canonical(self,tensor_set):
        left_canonical_tensor = [0 for i in range(self.num_dim + 1)]
        mat = tensor_set[0]
        mat = mat.reshape(-1, mat.shape[3])
        u, lamda, v = np.linalg.svd(mat, full_matrices=False)
        left_canonical_tensor[1] = np.dot(np.diag(lamda), v)
        for i in range(1,self.num_dim-1):
            mat = np.tensordot(left_canonical_tensor[i], tensor_set[i],[1,0])
            mat = mat.reshape(-1, mat.shape[-1])
            u,lamda,v = np.linalg.svd(mat, full_matrices=False)
            left_canonical_tensor[i+1] = np.dot(np.diag(lamda), v)
        return left_canonical_tensor

    def right_canonical(self, tensor_set):
        """
        Calculate the right tensor canonical for MPO format required
        :param left_tensor: the tensor_set output from function: left_canonical
        :return: the right_tensor_canonical format for calculate the mpo decomposition
        """
        right_canonical_tensor = [0 for i in range(self.num_dim + 1)]
        # print(tensor_set.shape)
        mat = tensor_set[self.num_dim - 1]
        mat = mat.reshape(mat.shape[0], -1)
        u, lamda, v = np.linalg.svd(mat, full_matrices=False)
        right_canonical_tensor[self.num_dim - 1] = np.dot(u, np.diag(lamda))

        for i in range(self.num_dim - 2, 0, -1):
            mat = np.tensordot(tensor_set[i], right_canonical_tensor[i + 1], [3, 0])
            mat = mat.reshape(mat.shape[0], -1)
            u, lamda, v = np.linalg.svd(mat, full_matrices=False)
            right_canonical_tensor[i] = np.dot(u, np.diag(lamda))
        return right_canonical_tensor

    def expectrum_normalization(self, lamda):
        """
        Do the lamda normalization for calculate the needed rank for MPO structure
        :param lamda: lamda parameter from left canonical
        :return:
        """
        norm_para = np.sum(lamda ** 2) ** (0.5)
        lamda_n = lamda / norm_para
        lamda_12 = lamda ** (-0.5)
        return lamda_n, np.diag(lamda_12)

    def gauge_aux_p_q(self, left_canonical_tensor, right_canonical_tensor):
        p = [0 for i in range(self.num_dim + 1)]
        q = [0 for i in range(self.num_dim + 1)]
        lamda_set = [0 for i in range(self.num_dim + 1)]
        lamda_set_value = [0 for i in range(self.num_dim + 1)]
        lamda_set[0] = np.ones([1,1])
        lamda_set[-1] = np.ones([1,1])
        for i in range(1, self.num_dim):
            mat = np.dot(left_canonical_tensor[i],right_canonical_tensor[i])
            # mat = right_canonical_tensor[i]
            u, lamda, v = np.linalg.svd(mat)
            lamda_n, lamda_l2 = self.expectrum_normalization(lamda)
            lamda_set[i] = lamda_n
            lamda_set_value[i] = lamda
            p[i] = np.dot(right_canonical_tensor[i], v.T)
            p[i] = np.dot(p[i],lamda_l2)
            q[i] = np.dot(lamda_l2,u.T)
            q[i] = np.dot(q[i], left_canonical_tensor[i])
        return p, q, lamda_set, lamda_set_value

    def mpo_canonical(self, tensor_set, p, q):
        tensor_set[0] = np.tensordot(tensor_set[0], p[1], [3,0])
        tensor_set[-1] = np.tensordot(q[self.num_dim-1], tensor_set[-1], [1,0])
        for i in range(1, self.num_dim-1):
            tensor_set[i] = np.tensordot(q[i],tensor_set[i],[1,0])
            tensor_set[i] = np.tensordot(tensor_set[i],p[i+1], [3,0])
        return tensor_set


    def truncated_tensor(self, tensor_set, step_train=False):
        """
        Get a untruncated tensor by mpo
        :param tensor_set: the input weight
        :return: a untruncated tensor_set by mpo
        """    
        if step_train:
            tensor_set_tmp = [i.detach().cpu().numpy() for i in tensor_set]
            cano_tensor_set = self.bi_canonical(tensor_set_tmp)
            tensor_set = torch.nn.ParameterList(
            [nn.Parameter(torch.from_numpy(i).cuda(), requires_grad=True) for i in cano_tensor_set])
            tensor_set[2].requires_grad = False

        mpo_trunc = self.mpo_truncate_ranks[:]
        for i in range(self.num_dim):
            if step_train:
                mask_noise = torch.ones_like(tensor_set[i])
            t = tensor_set[i]
            r_l = mpo_trunc[i]
            r_r = mpo_trunc[i + 1]
            if isinstance(tensor_set[i], nn.parameter.Parameter):
                if step_train:
                    
                    mask_noise[r_l:, :, :, :] = 0.0
                    mask_noise[:r_l, :, :, r_r:] = 0.0
                    tensor_set[i].data = tensor_set[i].data * mask_noise
                else:
                    tensor_set[i].data = t[:r_l, :, :, :r_r]
            else:
                tensor_set[i] = t[:r_l, :, :, :r_r]
                assert "Check! tensor_set is not nn.parameter.Parameter"
        return tensor_set

    def matrix2mpo(self, inp_matrix, cutoff=True):
        """
        Utilize the matrix to mpo format with or without cutoff
        :param inp_matrix: the input matrix, type=list
        :param cutoff: weather cut of not, type = bool
        :return: the truncated of not mps format of input matrix
        """
        tensor_set = self.get_tensor_set(inp_matrix)
        left_canonical_tensor = self.left_canonical(tensor_set)
        right_canonical_tensor = self.right_canonical(tensor_set)
        p,q,lamda_set, lamda_set_value = self.gauge_aux_p_q(left_canonical_tensor,right_canonical_tensor)
        tensor_set = self.mpo_canonical(tensor_set,p,q)
        if cutoff != False:
            tensor_set = self.truncated_tensor(tensor_set)
        return tensor_set,lamda_set, lamda_set_value
    def bi_canonical(self, tensor_set):
        left_canonical_tensor = self.left_canonical(tensor_set)
        right_canonical_tensor = self.right_canonical(tensor_set)
        p,q,_, _ = self.gauge_aux_p_q(left_canonical_tensor,right_canonical_tensor)
        tensor_set = self.mpo_canonical(tensor_set,p,q)

        return tensor_set
    def mpo2matrix(self, tensor_set):
        """
        shirnk the bond dimension to tranfer an mpo format to matrix format
        :param tensor_set: the input mpo format
        :return: the matrix format
        """
        t = tensor_set[0]
        # print(t.shape, tensor_set[1].shape)
        for i in range(1, self.num_dim):
            t = torch.tensordot(t, tensor_set[i], ([len(t.shape)-1],[0]))
        # Squeeze the first and the last 1 dimension
        t = t.squeeze(0)
        t = t.squeeze(-1)
        # Caculate the new index for mpo
        tmp1 = torch.tensor(range(len(self.mpo_output_shape))) * 2
        tmp2 = tmp1 + 1
        new_index = torch.cat((tmp1, tmp2), 0)
        # Transpose and reshape to output
        t = t.permute(tuple(new_index))
        t = t.reshape(torch.prod(torch.tensor(self.mpo_input_shape)),torch.prod(torch.tensor(self.mpo_output_shape)))
        return t

    def calculate_total_mpo_param(self, cutoff=True):
        # print("use cutoff: ", cutoff)
        total_size = 0
        if cutoff:
            rank = self.mpo_truncate_ranks
        else:
            rank = self.mpo_ranks
        for i in range(len(self.mpo_input_shape)):
            total_size += rank[i] * self.mpo_input_shape[i] * self.mpo_output_shape[i] * rank[i + 1]

        return total_size
    def new_mpo2matrix(self, tensor_set):
        """
        shirnk the bond dimension to tranfer an mpo format to matrix format
        :param tensor_set: the input mpo format
        :return: the matrix format
        """
        t = tensor_set[0]
        # print(t.shape, tensor_set[1].shape)
        for i in range(1, self.num_dim):
            t = torch.tensordot(t, tensor_set[i], ([len(t.shape)-1],[0]))
        t = t.reshape(torch.prod(torch.tensor(self.mpo_input_shape)),torch.prod(torch.tensor(self.mpo_output_shape)))
        return t
    @staticmethod
    def test_difference(matrix1, matrix2):
        """
        we input an matrix , return the difference between those two matrix
        :param matrix:
        :return:
        """
        v = matrix1 - matrix2
        error = np.linalg.norm(v)
        return error

class Experts(nn.Module):
    def __init__(self,
        dim=768,
        config=config,
        num_experts = 16,
        hidden_dim = None,
        activation = GELU,
        pdropout = 0.0,
        use_mpo = True,
        tensor_learn = False):
        super().__init__()
        
        self.use_mpo = use_mpo
        self.tensor_learn = tensor_learn
        self.config = config
        self.num_experts = num_experts
        hidden_dim = default(hidden_dim, dim * 4)
        num_experts = cast_tuple(num_experts)

        w1 = torch.zeros(*num_experts, dim, hidden_dim)
        w2 = torch.zeros(*num_experts, hidden_dim, dim)
        b1 = torch.zeros(hidden_dim)
        b2 = torch.zeros(dim)

        w1 = init_(w1)
        w2 = init_(w2)

        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        
        if self.use_mpo:
            if dim == 768:
                mpo_input_shape_fc, mpo_output_shape_fc, truncate_num_fc = [4,4,8,6,4],[3,4,4,4,4], config.linear_trunc # 768*3072
                mpo_input_shape_proj, mpo_output_shape_proj, truncate_num_proj =  [3,4,4,4,4], [4,4,8,6,4], config.linear_trunc # 3072*768
            elif config.n_embd == 1024:
                mpo_input_shape_fc, mpo_output_shape_fc, truncate_num_fc = [4,4,8,8,4],  [4,4,4,4,4],config.linear_trunc # 1024*4096
                mpo_input_shape_proj, mpo_output_shape_proj, truncate_num_proj =  [4,4,4,4,4], [4,4,8,8,4],config.linear_trunc # 4096*1024
            else:
                raise NotImplementedError
            self.mpo_w1 = MPO(mpo_input_shape_fc, mpo_output_shape_fc, truncate_num_fc)
            w1_tensort_set = [self.mpo_w1.matrix2mpo(w1[i].squeeze(0).numpy())[0] for i in range(self.num_experts)]
            self.w1_mpo = [np.stack([w1_tensort_set[i][j] for i in range(self.num_experts)], axis=0) for j in range(5)]
            
            self.mpo_w2 = MPO(mpo_input_shape_proj, mpo_output_shape_proj, truncate_num_proj)
            w2_tensort_set = [self.mpo_w2.matrix2mpo(w2[i].squeeze(0).numpy())[0] for i in range(self.num_experts)]
            self.w2_mpo = [np.stack([w2_tensort_set[i][j] for i in range(self.num_experts)], axis=0) for j in range(5)]
            # if 'mlp' in config.load_layer:
            self.convert_types()
            
        else:
            self.w2_mpo = None
            self.w1_mpo = None

        self.b1 = nn.Parameter(b1)
        self.b2 = nn.Parameter(b2)
        if not isfunction(activation):
            self.act = activation()
        else:
            self.act = activation
        
        self.dropout = nn.Dropout(pdropout)

    def forward(self, x):
        if self.use_mpo:
            for idx in range(5):
                print(self.w1_mpo[idx][0].shape)
            # i = 0
            # print(self.mpo_w1.mpo2matrix([self.w1_mpo[0][i], self.w1_mpo[1][i], self.w1_mpo[2],self.w1_mpo[3][i], self.w1_mpo[4][i]]))


            w1_rebuild = torch.stack([self.mpo_w1.mpo2matrix([self.w1_mpo[0][i], self.w1_mpo[1][i], self.w1_mpo[2],
                                                              self.w1_mpo[3][i], self.w1_mpo[4][i]]) for i in range(self.num_experts)],0)
            w2_rebuild = torch.stack([self.mpo_w2.mpo2matrix([self.w2_mpo[0][i], self.w2_mpo[1][i], self.w2_mpo[2],
                                                              self.w2_mpo[3][i], self.w2_mpo[4][i]]) for i in range(self.num_experts)],0)
            ######## 上面这个需要调换 d=768 h=3072
            hidden = torch.einsum('bnd,ehd->benh', x.to(w1_rebuild.device), w1_rebuild) + self.b1.to(w1_rebuild.device)
            hidden = self.act(hidden)
            out    = torch.einsum('benh,edh->bend', hidden, w2_rebuild) + self.b2.to(w1_rebuild.device)
        else:
            hidden = torch.einsum('...nd,...dh->...nh', x, self.w1) + self.b1
            hidden = self.act(hidden)
            out    = torch.einsum('...nh,...hd->...nd', hidden, self.w2) + self.b2
        return self.dropout(out)
    def convert_types(self):
        device = torch.cuda.current_device()
        for i in range(5):
            self.w1_mpo[i] = nn.Parameter(torch.from_numpy(self.w1_mpo[i]).to(device), requires_grad=True)
            self.w2_mpo[i] = nn.Parameter(torch.from_numpy(self.w2_mpo[i]).to(device), requires_grad=True)
        self.w1_mpo = nn.ParameterList(self.w1_mpo)
        self.w2_mpo = nn.ParameterList(self.w2_mpo)
        if self.tensor_learn:
            self.w1_mpo[2].requires_grad = False
            self.w2_mpo[2].requires_grad = False
        self.w1_mpo[2] = self.w1_mpo[2][0]
        self.w2_mpo[2] = self.w2_mpo[2][0]

    def from_pretrained_mpo(self, expert_new):
        logger.info("Check from_pretrained in Experts")
        device = torch.cuda.current_device()
        # 分配的数值对象会一直存在，但是变量名由于是在函数中则会消失
        ######## 下面这个需要调换
        # shared_c_fc_ct = expert_new.c_fc_mpo.tensor_set[2].clone().detach().cpu().numpy()
        # shared_c_proj_ct = expert_new.c_proj_mpo.tensor_set[2].clone().detach().cpu().numpy()
        
        # for i in range(self.num_experts):
        #     self.w1_mpo[2][i] = shared_c_fc_ct
        #     self.w2_mpo[2][i] = shared_c_proj_ct
        # TODO：下面的是正确的方法
        self.w1_mpo[2] = self.w1_mpo[2][0]
        self.w2_mpo[2] = self.w2_mpo[2][0]
        ######## 下面这个需要调换
        for j in [0,1,3,4]:
            for i in range(self.num_experts):
                self.w1_mpo[j][i] = expert_new.wi_mpo.tensor_set[j].clone().detach().cpu().numpy()
                self.w2_mpo[j][i] = expert_new.wo_mpo.tensor_set[j].clone().detach().cpu().numpy()

            self.w1_mpo[j] = nn.Parameter(torch.from_numpy(self.w1_mpo[j]).to(device), requires_grad=True)
            self.w2_mpo[j] = nn.Parameter(torch.from_numpy(self.w2_mpo[j]).to(device), requires_grad=True)
        
        self.w1_mpo[2] = nn.Parameter(torch.from_numpy(self.w1_mpo[2]).to(device), requires_grad=True)
        self.w2_mpo[2] = nn.Parameter(torch.from_numpy(self.w2_mpo[2]).to(device), requires_grad=True)
        self.w1_mpo = nn.ParameterList(self.w1_mpo)
        self.w2_mpo = nn.ParameterList(self.w2_mpo)

        if self.tensor_learn:
            for ind in self.config.train_AT_index:
                self.w1_mpo[ind].requires_grad = False
                self.w2_mpo[ind].requires_grad = False

# {
# 	"name": "",
# 	"message": "",
# 	"stack": "Running cells with 'mpo' requires the ipykernel package.
# Run the following command to install 'ipykernel' into the Python environment. 
# Command: 'conda install -n mpo ipykernel --update-deps --force-reinstall'"
# }
# 4096 11008
# w1 4096 > 24
# w2 24 >11008
W1_shape = (4096, 24)
# W2_shape = (24, 11008)
W2_shape = (24, 14336)
w1 = torch.zeros(W1_shape)
w2 = torch.zeros(W2_shape)
w1 = init_(w1)
w2 = init_(w2)
# att   4096-24-4096
mpo_input_shape_fc, mpo_output_shape_fc, truncate_num_fc = [4, 4, 16, 4, 4],[2,2,3,2,1], 10**5 
# mpo_input_shape_fc, mpo_output_shape_fc, truncate_num_fc = [4, 4, 16, 4, 4],[2,2,2,2,1], 10**5 
# mpo_input_shape_proj, mpo_output_shape_proj, truncate_num_proj = [2,2,3,2,1], [4, 4, 43, 4, 4], 10**5 
mpo_input_shape_proj, mpo_output_shape_proj, truncate_num_proj = [2,2,3,2,1], [4, 4, 56, 4, 4], 10**5 

# 4 * 4 * 56 * 4 * 4 
mpo_w1 = MPO(mpo_input_shape_fc, mpo_output_shape_fc, truncate_num_fc)

w1_tensort_set = mpo_w1.matrix2mpo(w1.squeeze(0).numpy())[0]

for x in w1_tensort_set:
    num = 1
    for n in x.shape:
        num *= n
    print('mpo_tenosr para num :',num)
    

print(mpo_w1.calculate_total_mpo_param(cutoff=False))
mpo_w2 = MPO(mpo_input_shape_proj, mpo_output_shape_proj, truncate_num_proj)
w2_tensort_set = mpo_w2.matrix2mpo(w2.squeeze(0).numpy())[0]
total = 0
for x in w2_tensort_set:
    num = 1
    for n in x.shape:
        num *= n
    print('mpo_tenosr para num :',num)
print(mpo_w2.calculate_total_mpo_param(cutoff=False))