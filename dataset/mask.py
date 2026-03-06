import numpy as np
    
    
def random_seismic_sampler_1(time, trace, seed, p=0.1):
    """_summary_

    Args:
        time (_type_): 时间采样个数
        trace (_type_): 道集个数
        seed (_type_): 随机种子
        p (float, optional): 缺失道集比例. Defaults to 0.1.

    Returns:
        _type_: _description_
    """
    #------随机去除道集的函数-------------------

    np.random.seed(seed)
    unif_random_matrix = np.random.uniform(0., 1., size = [trace])
    ans = np.ones((time,trace))
    for j in range(trace):
        ans[:,j] = unif_random_matrix[j]

    binary_random_matrix = 1-1*(ans< p)

    return binary_random_matrix


def continuous_seismic_sampler_1(time, trace,seed):

#------随机去除连续道集的函数-------------------

    np.random.seed(seed+1)
    unif_random = np.random.uniform(0.1, 0.2) # 即10%到20%缺失

    flag = np.random.uniform(0.0, 1.0, size = [1])
    
    if flag>0.5:
        np.random.seed(seed+1)
        start = np.random.randint(int(0.05*trace), int(0.1*trace))
    else:
        np.random.seed(seed+1)
        start = np.random.randint(int(0.6*trace), int(0.8*trace))
    end = (start + trace * unif_random).astype(np.int16)
    
    ans = np.ones([time,trace])
    


    ans[:,start:end+1] = 0
    binary_continus_matrix = ans.astype(float)
    
    return binary_continus_matrix

def Random_seismic_sampler(n,time,trace,p,flag=False):
    #------随机去除道集的函数-------------------   
    ans = np.ones([n,time,trace])
    ratio = []
    for i in range(n):
        seed =i
        if flag:
            np.random.seed(seed)
        # 计算每个样本中应该缺失的最小和最大列数
        min_missing_traces = int(np.ceil(0.2 * trace))
        max_missing_traces = int(np.floor(p * trace))
        # 从[min_missing_traces, max_missing_traces]范围内随机选择一个值作为缺失列数
        num_missing_traces = np.random.randint(min_missing_traces, max_missing_traces + 1)
        # 随机选择列进行缺失
        missing_indices = np.random.choice(trace, num_missing_traces, replace=False)
        ans[i,:,missing_indices] = 0
        ratio.append(np.sum(ans[i]==0)/(time*trace))
    min_ritio = np.min(np.array(ratio)) 
    max_ritio = np.max(np.array(ratio))        
    binary_random_matrix = ans.astype(float)
    return binary_random_matrix,min_ritio,max_ritio

def continus_seismic_sampler(n,time,trace,p,flag=False):
#------下面写随机去除连续道集的函数-------------------
    ans = np.ones([n,time,trace])
    ratio = []
    for i in range(n):
        seed = i
        if flag:
            np.random.seed(seed)
        unif_random = np.random.uniform(0.1, p, size = [1])
        start = np.random.randint(int(0.1*trace), int((1.0-p-0.1)*trace), size = [1])
        end = (start+trace*unif_random).astype(np.int16)
        star =start[0]
        en = end[0]
        ans[i,:,star:en+1]=0
        ratio.append(np.sum(ans[i]==0)/(time*trace))
    min_ritio = np.min(np.array(ratio)) 
    max_ritio = np.max(np.array(ratio))
    binary_random_matrix = ans.astype(float)
    return binary_random_matrix,min_ritio,max_ritio



if __name__ == '__main__':
    
    pass