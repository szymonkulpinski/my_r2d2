import torchvision
import numpy as np
import torch

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib import pyplot as pl;
pl.ion()
from scipy.ndimage import uniform_filter
smooth = lambda arr: uniform_filter(arr, 3)

def add_images_tensorboard(pred, writer, model_name, epoch):
    if model_name == 'r2d2':
        for key in pred.keys():
            if key != 'descriptors':
                img_grid = torchvision.utils.make_grid(pred[key][0], normalize=True)
                img_grid2 = torchvision.utils.make_grid(pred[key][1], normalize=True)
                combined_grid = torch.cat((img_grid, img_grid2), 1)
                writer.add_image(key, combined_grid, epoch)

                # img0 = img_grid.cpu().numpy().transpose(1, 2, 0)
                # img1 = not_norm.cpu().numpy().transpose(1, 2, 0)
                # print("sth")

        new_pred = get_one_batch(pred)
        writer.add_image("visual", get_image_without_keypoints(new_pred), epoch)
        writer.close()
    return

def get_one_batch(pred):
    new_pred ={}
    for key in pred.keys():
        new_pred[key] = pred[key][0][0:1] # so that it still keeps dim 1 of the tensor and not only image
    new_pred['imgs'] = torchvision.utils.make_grid(new_pred['imgs'], normalize=True)
    return new_pred

def transparent(img, alpha, cmap, **kw):
    from matplotlib.colors import Normalize
    colored_img = cmap(Normalize(clip=True, **kw)(img))
    colored_img[:, :, -1] = alpha
    return colored_img

# from r2d2 viz:
def get_image_without_keypoints(pred):

    img0 = pred['imgs']
    img = img0.cpu().numpy()
    img = img.transpose(1,2,0) # passed already as 3 dim tensor, not 4

    with torch.no_grad():
        res = pred
        rela = res.get('reliability')
        repe = res.get('repeatability')

    fig = pl.figure("viz")
    kw = dict(cmap=pl.cm.RdYlGn, vmax=1)
    crop = (slice(20, -20 or 1),) * 2

    ax1 = pl.subplot(131)
    pl.imshow(img[crop], cmap=pl.cm.gray)
    pl.xticks(());
    pl.yticks(())

    pl.subplot(132)
    pl.imshow(img[crop], cmap=pl.cm.gray)
    pl.xticks(());
    pl.yticks(())
    c = repe[0, 0].cpu().detach().numpy()
    # pl.show()
    pl.imshow(transparent(smooth(c)[crop], 0.5, vmin=0, **kw))

    ax1 = pl.subplot(133)
    pl.imshow(img[crop], cmap=pl.cm.gray)
    pl.xticks(());
    pl.yticks(())
    rela = rela[0, 0].cpu().detach().numpy()
    pl.imshow(transparent(rela[crop], 0.5, vmin=0.9, **kw))
    # pl.show()

    pl.gcf().set_size_inches(9, 2.73)
    pl.subplots_adjust(0.01, 0.01, 0.99, 0.99, hspace=0.1)

    canvas = FigureCanvas(fig)
    canvas.draw()  # draw the canvas, cache the renderer

    width, height = fig.get_size_inches() * fig.get_dpi()
    width = int(width)
    height = int(height)
    output_img = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)

    return torch.tensor(output_img.transpose(2, 0, 1), dtype=torch.float)