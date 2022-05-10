def contain(p, shape):
    return 0 <= p[0] < shape[0] and 0 <= p[1] < shape[1] # 범위 이내이면 True, 아니면 False

def contain_pts(p, p1, p2):
    return p1[0] <= p[0] < p2[0] and p1[1] <= p[1] < p2[1]