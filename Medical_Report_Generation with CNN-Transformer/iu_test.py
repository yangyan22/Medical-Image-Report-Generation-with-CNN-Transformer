import argparse
from tqdm import tqdm
import torch
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.models import Model
from modules.metrics import compute_scores
import json
import os


def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='/media/camlab1/doc_drive/IU_data/IU_Y', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='/media/camlab1/doc_drive/IU_data/IU_Y/iu_annotation_R2Gen.json', help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'], help='dataset')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')  # 60 100
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')   # 3 10
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=8, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet50', help='the visual extractor to be used.')  # resnet18
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')  # 2048:resnet101 512:resnet18
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
    parser.add_argument('--n_gpu', type=int, default=1, help='the sample number per image.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='greedy', help='the sample methods to sample a report.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')
    parser.add_argument('--restore_dir', type=str, default='./results/iu_50/', help='the patch to save the models.')
    args = parser.parse_args()
    return args


def main():
    args = parse_agrs()
    tokenizer = Tokenizer(args)
    test_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=False)
    
    model_path = os.path.join(args.restore_dir, "current_checkpoint.pth")
    checkpoint = torch.load(model_path)
    model = Model(args, tokenizer)
    model.load_state_dict(checkpoint['state_dict'])
    print("Checkpoint loaded from epoch {}".format(checkpoint['epoch']))
    device = torch.device('cuda:0')
    
    if args.n_gpu > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
        model = model.to(device)
        model = torch.nn.DataParallel(model, device_ids=[0, 1])
        print("GPUs_Used: {}".format(args.n_gpu))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        model = model.to(device)
   
    model.eval()
    with torch.no_grad():
        test_gts, test_res = [], []
        with tqdm(desc='Epoch %d - Testing', unit='it', total=len(test_dataloader)) as pbar:
            for batch_idx, (images_id, images, reports_ids, reports_masks) in enumerate(test_dataloader):  # test
                images, reports_ids, reports_masks = images.to(device), reports_ids.to(device), reports_masks.to(device)
                output = model(images, mode='sample')
                if args.n_gpu > 1:
                    reports = model.module.tokenizer.decode_batch(output.cpu().numpy())
                    ground_truths = model.module.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                else:
                    reports = model.tokenizer.decode_batch(output.cpu().numpy())
                    ground_truths = model.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())
                pbar.update()
                test_res.extend(reports)
                test_gts.extend(ground_truths)
                # print("Report IDS")
                # print(images_id)
                # print("Generated Reports")
                # print(reports)
                # print("Ground_Truths Reports")
                # print(ground_truths)
                i = 0
                for id in images_id:
                    print("Report IDS: {}".format(id))
                    print('Pred Sent: {}'.format(reports[i]))
                    print('Real Sent: {}'.format(ground_truths[i]))
                    print('\n')
                    i = i + 1
        test_met = compute_scores({i: [gt] for i, gt in enumerate(test_gts)},
                                    {i: [re] for i, re in enumerate(test_res)})
        print(test_met)
        with open(os.path.join(args.restore_dir, 'data_gt.json'), 'w') as f:
            json.dump(test_gts, f)
        with open(os.path.join(args.restore_dir, 'data_re.json'), 'w') as f:
            json.dump(test_res, f)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
