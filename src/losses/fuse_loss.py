import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from src.losses.registry import fuseloss
from src.utils.tool import create_file


def SRDB(v_logit, i_logit, label, zero_t):
    """
    Compute semantic region masks based on logits and labels.

    Args:
        vi_logit (torch.Tensor): Logit tensor for the visible image.
        ir_logit (torch.Tensor): Logit tensor for the infrared image.
        label (torch.Tensor): Ground truth label tensor.
        zero_tensor (torch.Tensor): All-zero tensor of the same shape as the label.

    Returns:
        torch.Tensor: Background mask (M_bg).
        torch.Tensor: Correctly segmented visible and infrared mask (M_VT_IT).
        torch.Tensor: Incorrectly segmented visible mask (M_VF_IT).
        torch.Tensor: Incorrectly segmented infrared mask (M_VT_IF).
        torch.Tensor: Incorrectly segmented visible and infrared mask (M_VF_IF).
    """
    M_bg = torch.eq(zero_t, label)
    M_pre = torch.ne(zero_t, label)
    
    M_VT = torch.eq(v_logit, label)
    M_VF = torch.ne(v_logit, label)
    M_IT = torch.eq(i_logit, label)
    M_IF = torch.ne(i_logit, label)

    M_VT_IT = M_VT * M_IT * M_pre
    M_VT_IF = M_VT * M_IF * M_pre
    M_VF_IT = M_VF * M_IT * M_pre
    M_VF_IF = M_VF * M_IF * M_pre
    
    return M_bg, M_VT_IT, M_VT_IF, M_VF_IT, M_VF_IF


@fuseloss.register
class SRAFLoss(nn.Module):
    def __init__(self):
        super(SRAFLoss, self).__init__()
        self.sobelConv = Sobelxy()
        self.l2loss = nn.MSELoss()

        self.alpha = 10
        self.beta = 10
        self.gama = 8
        self.delta = 4

    def forward(self, image_vis_ycrcb, image_ir, fused_y, vi_logit, ir_logit, label, img_name, epoch, save_pic):
        image_y = image_vis_ycrcb[:, :1, :, :]
        b, _, h, w = image_y.shape
        y_grad = self.sobelConv(image_y)
        ir_grad = self.sobelConv(image_ir)
        generate_img_grad = self.sobelConv(fused_y)
        w_gra_vi = torch.ge(y_grad, ir_grad)
        w_gra_ir = torch.ge(ir_grad, y_grad)

        x_in_max = torch.max(image_y, image_ir)
        x_grad_max = torch.max(y_grad, ir_grad)

        zero_tensor = torch.zeros_like(image_ir)
        M_bg, M_VT_IT, M_VT_IF, M_VF_IT, M_VF_IF = SRDB(v_logit=vi_logit, i_logit=ir_logit, label=label, zero_t=zero_tensor)

        # Areas lacking high-level semantic information (M_bg and M_VF_IF)
        x_ins_max_lac = torch.max((M_bg+M_VF_IF) * image_y, (M_bg+M_VF_IF) * image_ir)
        loss_ins_lac = F.l1_loss((M_bg+M_VF_IF) * fused_y, x_ins_max_lac)
        x_gra_max_lac = torch.max((M_bg+M_VF_IF) * y_grad, (M_bg+M_VF_IF) * ir_grad)
        loss_gra_lac = F.l1_loss((M_bg+M_VF_IF) * generate_img_grad, x_gra_max_lac)
        loss_total_lac = self.alpha * loss_ins_lac + self.beta * loss_gra_lac

        # Regions of single-modality semantics (M_VF_IT and M_VT_IF)
        loss_ins_ps_TF = self.l2loss(M_VT_IF * image_y, M_VT_IF * fused_y)
        loss_gra_ps_TF = self.l2loss(M_VT_IF * generate_img_grad, M_VT_IF * y_grad)
        loss_ins_ps_FT = self.l2loss(M_VF_IT * image_ir, M_VF_IT * fused_y)
        loss_gra_ps_FT = self.l2loss(M_VF_IT * generate_img_grad, M_VF_IT * ir_grad)
        loss_total_uni = self.gama * (loss_ins_ps_TF+loss_ins_ps_FT + self.beta * (loss_gra_ps_TF+loss_gra_ps_FT))

        # Semantically rich regions (M_VT_IT)
        x_ins_max_TT = torch.max(M_VT_IT * w_gra_vi * image_y, M_VT_IT * w_gra_ir * image_ir)
        loss_ins_ps_TT = F.l1_loss(M_VT_IT * fused_y, M_VT_IT * x_ins_max_TT)
        x_gra_max_TT = torch.max(M_VT_IT * y_grad, M_VT_IT * ir_grad)
        loss_gra_ps_TT = F.l1_loss(M_VT_IT * x_gra_max_TT, M_VT_IT * generate_img_grad)
        loss_total_bi = self.gama * (loss_ins_ps_TT + self.beta * loss_gra_ps_TT)

        if (epoch % 10 == 0 or epoch == 1) and save_pic:
        # if (epoch == 1) and save_pic:
            create_file(r'./visualization/save_pic/')
            create_file(r'./visualization/save_pic/' + 'epoch_{}'.format(epoch))
            """
            self._save_mask(tensor=M_VT_IT, img_name=img_name, tso_name='Mask_bi', epoch=epoch)
            self._save_mask(tensor=M_VT_IF, img_name=img_name, tso_name='Mask_vtif', epoch=epoch)
            self._save_mask(tensor=M_VF_IT, img_name=img_name, tso_name='Mask_vfit', epoch=epoch)
            self._save_mask(tensor=M_bg + M_VF_IF, img_name=img_name, tso_name='Mask_lac', epoch=epoch)

            self._save_img(tensor=x_ins_max_lac, img_name=img_name, tso_name='Img_ins_lac', epoch=epoch)
            self._save_gra(tensor=x_gra_max_lac, img_name=img_name, tso_name='Img_gra_lac', epoch=epoch)
            self._save_img(tensor=M_VT_IF * image_y, img_name=img_name, tso_name='Img_ins_vtif', epoch=epoch)
            self._save_gra(tensor=M_VT_IF * y_grad, img_name=img_name, tso_name='Img_gra_vtif', epoch=epoch)
            self._save_img(tensor=M_VF_IT * image_ir, img_name=img_name, tso_name='Img_ins_vfit', epoch=epoch)
            self._save_gra(tensor=M_VF_IT * ir_grad, img_name=img_name, tso_name='Img_gra_vfit', epoch=epoch)
            self._save_img(tensor=x_ins_max_TT, img_name=img_name, tso_name='Img_ins_bi', epoch=epoch)
            self._save_gra(tensor=x_gra_max_TT, img_name=img_name, tso_name='Img_gra_bi', epoch=epoch)

            self._save_img(tensor=x_ins_max_lac+M_VT_IF * image_y+M_VF_IT * image_ir+x_ins_max_TT, img_name=img_name, tso_name='grdt_ins', epoch=epoch)
            self._save_gra(tensor=x_gra_max_lac+M_VT_IF * y_grad+M_VF_IT * ir_grad+x_gra_max_TT, img_name=img_name, tso_name='grdt_gra', epoch=epoch)
            self._save_img(tensor=x_in_max, img_name=img_name, tso_name='x_ins_max', epoch=epoch)
            self._save_gra(tensor=x_grad_max, img_name=img_name, tso_name='x_grad_max', epoch=epoch)
            """
            # self._save_mask(tensor=w_gra_vi, img_name=img_name, tso_name='w_gra_vi', epoch=epoch)
            # self._save_mask(tensor=w_gra_ir, img_name=img_name, tso_name='w_gra_ir', epoch=epoch)
        loss_total = loss_total_bi + self.delta * loss_total_uni + loss_total_lac

        return loss_total, loss_total_bi, loss_total_uni, loss_total_lac

    def _save_mask(self, tensor, img_name, tso_name, epoch):
        # Create directories for saving mask and img
        save_path = r'./visualization/save_pic/' + 'epoch_{}'.format(epoch) + '/' + tso_name
        create_file(save_path)

        # Get the dimensions of the tensor maps
        batch_size, channels, height, width = tensor.size()

        for i in range(batch_size):
            # Get the i-th tensor map
            # tensor_map = tensor[i]
            # tensor_map = tensor_map.unsqueeze(1)
            image_name = "{}".format(img_name[i])
            save_file = os.path.join(save_path, image_name)
            # grid = make_grid(tensor_map, nrow=int(channels ** 0.5))
            save_image(tensor[i].float(), save_file)

    def _save_gra(self, tensor, img_name, tso_name, epoch):
        # Create directories for saving mask and img
        save_path = r'./visualization/save_pic/' + 'epoch_{}'.format(epoch) + '/' + tso_name
        create_file(save_path)

        # Get the dimensions of the tensor maps
        batch_size, channels, height, width = tensor.size()

        for i in range(batch_size):
            # Get the i-th tensor map
            image_name = "{}".format(img_name[i])
            save_file = os.path.join(save_path, image_name)
            # grid = make_grid(tensor_map, nrow=int(channels ** 0.5))
            save_image(tensor[i].float(), save_file)

    def _save_img(self, tensor, img_name, tso_name, epoch):
        # Create directories for saving mask and img
        save_path = r'./visualization/save_pic/' + 'epoch_{}'.format(epoch) + '/' + tso_name
        create_file(save_path)

        # Get the dimensions of the tensor maps
        batch_size, channels, height, width = tensor.size()

        for i in range(batch_size):
            # Get the i-th tensor map
            tensor_map = (tensor[i] - tensor[i].min()) / (tensor[i].max() - tensor[i].min())
            # tensor_map = tensor[1]
            image_name = "{}".format(img_name[i])
            save_file = os.path.join(save_path, image_name)
            save_image(tensor_map.float(), save_file)


@fuseloss.register
class WoSRAFLoss(nn.Module):
    def __init__(self):
        super(WoSRAFLoss, self).__init__()
        self.sobelconv = Sobelxy()

    def forward(self, image_vis, image_ir, generate_img):
        image_y = image_vis[:, :1, :, :]
        x_in_max = torch.max(image_y, image_ir)
        loss_in = F.l1_loss(x_in_max, generate_img)

        y_grad = self.sobelconv(image_y)
        ir_grad = self.sobelconv(image_ir)
        generate_img_grad = self.sobelconv(generate_img)
        x_grad_joint = torch.max(y_grad, ir_grad)
        loss_grad = F.l1_loss(x_grad_joint, generate_img_grad)

        loss_total = loss_in + 10 * loss_grad

        return loss_total, loss_in, loss_grad


class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)


if __name__ == '__main__':
    vi = torch.randint(0, 2, (2, 3, 480, 640)).float()
    ir = torch.randint(0, 2, (2, 1, 480, 640)).float()
    vi = vi.cuda()
    ir = ir.cuda()
    vi_logit = torch.randint(0, 2, (2, 1, 480, 640)).float()
    ir_logit = torch.randint(0, 2, (2, 1, 480, 640)).float()
    label = torch.randint(0, 2, (2, 1, 480, 640)).float()
    img_name = ['1.png', '2.png']
    epoch = 0

    vi_logit = vi_logit.cuda()
    ir_logit = ir_logit.cuda()
    label = label.cuda()

    fused = torch.randint(0,2,(2, 1, 480, 640)).float()
    fused = fused.cuda()

    f_loss = SRAFLoss()
    loss_t, bi, uni, lac = f_loss(vi, ir, fused, vi_logit, ir_logit, label, img_name, epoch)
    print(loss_t)
