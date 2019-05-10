# fast_classify.py
import argparse
import os, sys
from inception3_classifier import tf, classify_batch, maybe_download_and_extract

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
if __name__ == '__main__':
    # handle incoming arguments
    parser = argparse.ArgumentParser(description="Classify Files with Inception3 base model")
    
    # handle image directory
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help='path to root of image directory, can contain subdirectories \
              with images, the program will search for all images and \
              predict them')
    
    # handle results file name
    parser.add_argument(
            "--results_file", type=str, required=True,
            help='path to the file to which to store the predictions')
    
    # batch size
    parser.add_argument(
        "--batch_size", type=int, required=False, default=20,
        help='how many files to run in sub-batches, default=20')
    
    # num predictions per image
    parser.add_argument(
        "--predictions", type=int, required=False, default=1,
        help="number of predictions per image, default=1")
    
    parser.add_argument(
        '--verbose', type=str2bool, required=False, default=True,
        help='set verbose output, default=True')
                       
                       
    
    # gather argument dictionary
    args = vars(parser.parse_args())
    
        
    # call classifier
    data = classify_batch(
        image_dir=args['image_dir'],
        results_file=args['results_file'],
        batch_size=args['batch_size'],
        num_predictions=args['predictions'],
        verbose=args['verbose']
    )
    print("done.")