import pickle
import os
import numpy as np
from scipy.ndimage.measurements import label
from tool import evaluate
import argparse
from scipy.ndimage import convolve
import torch.nn.functional as F
import torch
import math


def video_label_length(dataset='DAD_Jigsaw'):
    label_path = "/home/work/Alpha/Jigsaw-VAD/DAD_Jigsaw/testing/frame_masks"
    video_length = {}
    files = sorted(os.listdir(label_path))
    for f in files:
        label = np.load("{}/{}".format(label_path, f))
        video_length[f.split(".")[0]] = label.shape[0]  # 각 파일의 프레임 길이를 저장
    return video_length


def score_smoothing(score, ws=43, function='mean', sigma=10):
    assert ws % 2 == 1, 'window size must be odd'
    assert function in ['mean', 'gaussian'], 'Invalid window function type.'

    r = ws // 2
    weight = np.ones(ws)
    for i in range(ws):
        if function == 'mean':
            weight[i] = 1. / ws
        elif function == 'gaussian':
            weight[i] = np.exp(-(i - r) ** 2 / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

    weight /= weight.sum()
    new_score = score.copy()
    new_score[r: score.shape[0] - r] = np.correlate(score, weight, mode='valid')
    return new_score


def load_frames(dataset, frame_num=7):
    root = '/home/work/Alpha/Jigsaw-VAD'
    data_dir = os.path.join(root, dataset, 'testing', 'top_depth')

    file_list = sorted(os.listdir(data_dir))
    frames_list = []
    videos_list = []
    start_ind = frame_num // 2

    for video_file in file_list:
        if video_file not in videos_list:
            videos_list.append(video_file)
        frame_list = os.listdir(os.path.join(data_dir, video_file))
        for frame in range(start_ind, len(frame_list) - start_ind):
            frames_list.append({"video_name": video_file, "frame": frame})

    print(f"Loaded {len(videos_list)} videos with {len(frames_list)} frames in total.")
    return frames_list


def remake_video_output(video_output, dataset='DAD_Jigsaw'):
    video_length = video_label_length(dataset=dataset)
    return_output_spatial = []
    return_output_temporal = []
    return_output_complete = []
    return_output_clips = []  # 클립 단위 평가 결과 저장

    video_l = sorted(video_output.keys())
    for video in video_l:
        frame_record = video_output[video]
        frame_l = sorted(frame_record.keys())
        video_spatial = np.ones(video_length[video])
        video_temporal = np.ones(video_length[video])
        clip_scores = []  # 클립 단위 점수를 저장할 리스트

        for fno in frame_l:
            clip_record = np.array(frame_record[fno])
            video_spatial[fno], video_temporal[fno] = clip_record.min(0)
            clip_scores.append(clip_record.mean(0))  # 클립 단위 평균 점수 계산

        # 프레임 단위 점수 정규화
        non_ones = (video_spatial != 1).nonzero()[0]
        video_spatial[non_ones] = (video_spatial[non_ones] - video_spatial[non_ones].min()) / \
                                  (video_spatial[non_ones].max() - video_spatial[non_ones].min())

        non_ones = (video_temporal != 1).nonzero()[0]
        video_temporal[non_ones] = (video_temporal[non_ones] - video_temporal[non_ones].min()) / \
                                   (video_temporal[non_ones].max() - video_temporal[non_ones].min())

        # 클립 단위 점수 정규화
        clip_scores = np.array(clip_scores)
        clip_scores[:, 0] = (clip_scores[:, 0] - clip_scores[:, 0].min()) / \
                            (clip_scores[:, 0].max() - clip_scores[:, 0].min())
        clip_scores[:, 1] = (clip_scores[:, 1] - clip_scores[:, 1].min()) / \
                            (clip_scores[:, 1].max() - clip_scores[:, 1].min())

        # 스무딩 처리
        video_spatial = score_smoothing(video_spatial)
        video_temporal = score_smoothing(video_temporal)
        clip_scores = np.mean(clip_scores, axis=1)  # 클립 점수를 평균 내어 사용

        # 결과 저장
        return_output_spatial.append(video_spatial)
        return_output_temporal.append(video_temporal)
        return_output_clips.append(clip_scores)
        combined_video = (video_spatial + video_temporal) / 2.0
        return_output_complete.append(combined_video)

    return return_output_spatial, return_output_temporal, return_output_complete, return_output_clips


def evaluate_auc(video_output, dataset='DAD_Jigsaw'):
    result_dict = {'dataset': dataset, 'psnr': video_output}
    
    # evaluate_all 호출로 프레임 및 클립 단위 결과를 모두 계산
    frame_result, frame_avg_result, clip_result = evaluate.evaluate_all(result_dict, reverse=True, smoothing=True)

    print("(smoothing: True): Frame AUC: {}, Avg Frame AUC: {}, Clip AUC: {}".format(
        frame_result.auc, frame_avg_result[0], clip_result.auc))
    
    return frame_result, frame_avg_result, clip_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anomaly Prediction')
    parser.add_argument('--file', default=None, type=str, help='pkl file')
    parser.add_argument('--dataset', default='DAD_Jigsaw', type=str)
    parser.add_argument('--frame_num', required=True, type=int)

    args = parser.parse_args()

    with open(args.file, 'rb') as f:
        output = pickle.load(f)

    # Process video outputs
    video_output_spatial, video_output_temporal, video_output_complete, video_output_clips = \
        remake_video_output(output, dataset=args.dataset)

    # Evaluate AUC
    evaluate_auc(video_output_spatial, dataset=args.dataset)
    evaluate_auc(video_output_temporal, dataset=args.dataset)
    evaluate_auc(video_output_complete, dataset=args.dataset)
    evaluate_auc(video_output_clips, dataset=args.dataset)
