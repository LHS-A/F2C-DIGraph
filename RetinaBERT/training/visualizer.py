# -- coding: utf-8 --
from visdom import Visdom
import numpy as np

class Visualizer():
    def __init__(self, env="default", **kwargs):  # **kwargs必须要有，目的是灵活添加各种参数如opts
        self.vis = Visdom(env=env, **kwargs)  # init初始化可以直接复制，也可以传参赋值
        self.index = {}  # 建立空字典，方便对不同win窗口数值调用以及修改

    # plot方法用于接力赛，在plot_line静态绘制基础上动态续绘制剩余点
    def plot(self, win, y, con_point,x=None, **kwargs):
        if x is not None:
            x = x
        else:
            x = self.index.get(win, con_point)  # ----dict.update(key,value)，类似于list中的list.append()
        # dict中get方法：如果在dict中含有key(win)，则返回该key的value，如果找不到就返回can_point
        self.vis.line(Y=np.array([y]), X=np.array([x]), win=str(win), update=None if x == 0 else "append", **kwargs)
        self.index[win] = x + 1  # 当前窗口值对应值加一

    # 用于静态绘制，值均取自于列表，一次性绘制完成，win与plot一致就可以动态续接绘制
    def plot_line(self, win, y, **kwargs):
        self.vis.line(win=win, X=np.linspace(1, len(y), len(y)), Y=y, **kwargs)

    def img(self, name, img_, **kwargs):
        # images可以绘制BCHW格式的图片，数量是B的个数，可以指定每行显示图片数
        self.vis.images(img_, win=str(name),  # 窗口名称
                        opts=dict(title=name),  # 图像名
                        **kwargs)
        
    def plot_pred_contrast(self,pred,label,image):
        self.img(name="image", img_= image)
        self.img(name="pred", img_= pred)
        self.img(name="label", img_= label)
    
    def plot_entropy(self,H):
        self.img(name="pred_entropy", img_= H)

    def plot_metrics_total(self, metrics_dict):
        """
        Function: 绘制全体指标曲线图，log文件查看具体类别指标!
        """
        for metric, values in metrics_dict.items():
            if not values:  # 检查列表是否为空
                print(f"Warning: No data available for metric '{metric}'. Skipping plotting.")
                continue
            
            # 绘制每个指标的曲线图
            self.plot(win=metric, y=values[-1], opts=dict(title=metric, xlabel="Epoch", ylabel=metric), con_point=len(values))
            


if __name__ == "__main__":
    env_name = "随便写" 
    vis = Visualizer(env=env_name, port=5678)
    vis.plot(win="train_loss", y=loss.item(), con_point=len(args.train_loss),
                opts=dict(title="Train_Loss", xlabel="batch", ylabel="train_loss"))