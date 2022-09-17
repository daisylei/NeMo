# Speaker Foreign Accent Recognition

## Performance
|       Test metric         | FAE-CV22 test (eval.tsv) | FAE-CV22 eval (eval0.tsv) |
|:-------------------------:|:------------------------:|:-------------------------:|
|     test_acc_macro        | 0.4180358350276947       | 0.3351203203201294        |
|      test_auroc           | 0.8158841729164124       | 0.7418038249015808        |
| test_epoch_accuracy_top@1 | 0.4737991392612457       | 0.4175257682800293        |
|       test_loss           | 7.080672264099121        | 8.503963470458984         |


### Fine Tuning
For fine tuning on a pretrained .nemo speaker recognition model,
```bash
python speaker_fae_reco.py --config-path conf/ --config-name ecapa_tdnn-finetune-cv.yaml model.train_ds.manifest_filepath=manifest-train.json  model.validation_ds.manifest_filepath=manifest-eval.json model.test_ds.manifest_filepath=manifest-test.json model.decoder.num_classes=11
```
for fine tuning tips see this [tutorial](https://github.com/schaltung/NeMo/blob/fae/cv22/tutorials/speaker_tasks/FAE_Speaker_Accent_Classification.ipynb)

## Inference
We provide generic scripts for manifest file creation, embedding extraction, FAE-CV22 evaluation.


### Manifest Creation
We first generate manifest file to get embeddings. The embeddings are then used by `faecv22_eval.py` script to get similarity scores based on cosine-distance.  

```bash
# create list of files from voxceleb1 test folder (40 speaker test set)
find <path/to/voxceleb1_test/directory/> -iname '*.wav' > test_files.txt
python <NeMo_root>/scripts/speaker_tasks/filelist_to_manifest.py --filelist test_files.txt --id -3 --out test_manifest.json 
```

### Embedding Extraction 
Now using the manifest file created, we can extract embeddings to `data` folder using:
```bash
python extract_accent_embeddings.py --manifest=test_manifest.json --model_path='??' --embedding_dir='./'
```

### FAE CV22 Evaluation
``` bash
python fae-cv22_eval.py --trial_file='/path/to/trail/file' --emb='./embeddings/voxceleb1_test_manifest_embeddings.pkl' 
``` 


### Accent label inference
Using data from an enrollment set, one can infer labels on a test set using various backends such as cosine-similarity or a neural classifier.

To infer speaker labels using cosine_similarity backend
```bash 
python speaker_identification_infer.py data.enrollment_manifest=<path/to/enrollment_manifest> data.test_manifest=<path/to/test_manifest> backend.backend_model=cosine_similarity
``` 
refer to conf/speaker_identification_infer.yaml for more options.

## FAE CV Data Preparation

Scripts we provide for data preparation are very generic and can be applied to any dataset with a few path changes. 


For VoxCeleb datasets, we first download the datasets individually and make a list of audio files. Then we use the script to generate manifest files for training and validation. 
Download [voxceleb1](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html) and [voxceleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html) data. 

Once downloaded and uncompressed, use programs such as ffmpeg to convert audio files from m4a format to wav format. 
Refer to the following sample command
```bash
ffmpeg -v 8 -i </path/to/m4a/file> -f wav -acodec pcm_s16le <path/to/wav/file> 
```

Generate a list file that contains paths to all the dev audio files from voxceleb1 and voxceleb2 using find command as shown below:
```bash 
find <path/to/voxceleb1/dev/folder/> -iname '*.wav' > voxceleb1_dev.txt
find <path/to/voxceleb2/dev/folder/> -iname '*.wav' > voxceleb2_dev.txt
cat voxceleb1_dev.txt voxceleb2_dev.txt > voxceleb12.txt
``` 

This list file is now used to generate training and validation manifest files using a script provided in `<NeMo_root>/scripts/speaker_tasks/`. This script has optional arguments to split the whole manifest file in to train and dev and also chunk audio files to smaller segments for robust training (for testing, we don't need this). 

```bash
python <NeMo_root>/scripts/speaker_tasks/filelist_to_manifest.py --filelist voxceleb12.txt --id -3 --out voxceleb12_manifest.json --split --create_segments
```
This creates `train.json, dev.json` in the current working directory.
