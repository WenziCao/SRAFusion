import os
import numpy as np
import torch

from torch.autograd import Variable
from sklearn.metrics import confusion_matrix

from src.datasets.build import build_MSRS_dataloader
from src.logger.build import build_logger
from src.models.build import build_isg_net
from src.utils.tool import visualize, create_file, compute_results

from src.test.registry import tester


@tester.register
def test_isg(cfg):
    # basic data
    conf_total = np.zeros((cfg.MODEL.SEG_NET.NUM_CLASSES, cfg.MODEL.SEG_NET.NUM_CLASSES))
    label_list = ["unlabeled", "car", "person", "bike", "curve",
                  "car_stop", "guardrail", "color_cone", "bump"]

    # saved model path
    save_model_path = os.path.join('./model_hub/', cfg.MODEL.SEG_NET.TYPEI,
                                   'cp-epoch-{}.pt'.format(cfg.TEST_EPOCH))
    # visual image dir

    # get the model
    model = build_isg_net(cfg)
    model.to(cfg.DEVICE)
    model.eval()
    checkpoint = torch.load(save_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    print('model load done!')

    # dataset
    test_loader = build_MSRS_dataloader(cfg)
    test_loader.n_iter = len(test_loader)
    logger = build_logger(cfg)

    # begin to train
    with torch.no_grad():
        for it, (images_vis, images_ir, labels, name) in enumerate(test_loader):

            images_ir = Variable(images_ir).to(cfg.DEVICE)
            labels = Variable(labels).to(cfg.DEVICE)

            # logits.size(): mini_batch*num_class*480*640
            logits = model(images_ir)

            # convert tensor to numpy 1d array, size: minibatch*640*480
            label = labels.cpu().numpy().squeeze().flatten()
            prediction = logits.argmax(1).cpu().numpy().squeeze().flatten()
            # generate confusion matrix frame-by-frame
            conf = confusion_matrix(y_true=label, y_pred=prediction,
                                    labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
            conf_total += conf
            # save demo images
            fpath = []
            lb_path = []
            create_file('./visualization/VL_i_logit')
            create_file('./visualization/VL_gt_label')
            for k in range(len(name)):
                fpath.append('./visualization/VL_i_logit/' + name[k])
                lb_path.append('./visualization/VL_gt_label/' + name[k])
            # visual label and logits
            visualize(image_dir=fpath, predictions=logits.argmax(1))
            visualize(image_dir=lb_path, predictions=labels)

    precision_per_class, recall_per_class, iou_per_class = compute_results(conf_total)
    message = '*'.join(['\n* recall per class:\n',
                        'unlabeled:{rc_unlabeled:.6f},car:{rc_car:.6f},person:{rc_person:.6f},'
                        'bike:{rc_bike:.6f},curve:{rc_curve:.6f},car_stop:{rc_car_stop:.6f},'
                        'guardrail:{rc_guardrail:.6f},color_cone:{rc_color_cone:.6f},bump:{rc_bump:.6f}',
                        '\n* iou per class:\n',
                        'unlabeled:{unlabeled:.6f},car:{car:.6f},person:{person:.6f},bike:{bike:.6f},curve:{curve:.6f},'
                        'car_stop:{car_stop:.6f},guardrail:{guardrail:.6f},color_cone:{color_cone:.6f},bump:{bump:.6f}',
                        '\n* average values (np.mean(x)): \n recall:{recall:.6f},iou:{iou:.6f}'
                        '\n* average values (np.mean(np.nan_to_num(x))): \n ntn_recall:{recall:.6f},ntn_iou:{iou:.6f}'
                        ]) \
        .format(rc_unlabeled=recall_per_class[0], rc_car=recall_per_class[1], rc_person=recall_per_class[2],
                rc_bike=recall_per_class[3], rc_curve=recall_per_class[4], rc_car_stop=recall_per_class[5],
                rc_guardrail=recall_per_class[6], rc_color_cone=recall_per_class[7], rc_bump=recall_per_class[8],
                unlabeled=iou_per_class[0], car=iou_per_class[1], person=iou_per_class[2],
                bike=iou_per_class[3], curve=iou_per_class[4], car_stop=iou_per_class[5],
                guardrail=iou_per_class[6], color_cone=iou_per_class[7], bump=iou_per_class[8],
                recall=recall_per_class.mean(), iou=iou_per_class.mean(),
                ntn_recall=np.mean(np.nan_to_num(recall_per_class)), ntn_iou=np.mean(np.nan_to_num(iou_per_class)))
    logger.info(message)
