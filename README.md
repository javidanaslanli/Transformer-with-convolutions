# Transformer with Convolutions

I used https://arxiv.org/abs/2103.15808 paper to implement transformer with convolutions. This architecture has a lot of advantages. In ViT models we need positional embeddings, but in CvT model we do not need positional embeddings because of the usage of convolutions. Also with the help of convolutions we can learn much more information about parts of images. I trained this model only one time and I reached 74.1% accuracy with https://www.kaggle.com/datasets/puneet6060/intel-image-classification dataset. 

### This is the architecture

<img width="1021" alt="Screen_Shot_2021-07-20_at_11 50 20_AM_1CVroUG" src="https://github.com/javidanaslanli/CvT---Transformers-with-convolutions-from-scratch/assets/145380543/e4604757-159b-47b1-9832-67456a817c0e">








