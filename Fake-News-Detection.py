import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
# nltk.download('stopwords')
# nltk.download('wordnet')
import os

#先定位到工程目录下
projectPath=os.path.dirname(os.path.dirname(__file__))
#csv文件的位置
train_Path=os.path.join(projectPath,"Fake-News-Detection/train.csv")
print("最终目录：", train_Path)

# load the dataset
#导入数据集
news_d = pd.read_csv(train_Path)
 
print("Shape of News data:", news_d.shape)
print("News data columns", news_d.columns)

#通过使用df.head()我们可以立即熟悉数据集。
news_d.head()

# 最小，平均值，最大和四分位范围
#文本文字统计
txt_length = news_d.text.str.split().str.len()
txt_length.describe()
#标题统计
title_length = news_d.title.str.split().str.len()
title_length.describe()

#类别分布
#两个标签的计数图
sns.countplot(x="label", data=news_d);
print("1: Unreliable")
print("0: Reliable")
print("Distribution of labels:")
print(news_d.label.value_counts());

print(round(news_d.label.value_counts(normalize=True),2)*100);

#数据清洗
# Constants that are used to sanitize the datasets
# 用于清除数据集的常量
column_n = ['id', 'title', 'author', 'text', 'label']
remove_c = ['id','author']
categorical_features = []
target_col = ['label']
text_f = ['title', 'text']

# Clean Datasets
import nltk
from nltk.corpus import stopwords
import re
from nltk.stem.porter import PorterStemmer
from collections import Counter
 
ps = PorterStemmer()
wnl = nltk.stem.WordNetLemmatizer()
 
stop_words = stopwords.words('english')
stopwords_dict = Counter(stop_words)
 
# Removed unused clumns
# 删除未使用的块
def remove_unused_c(df,column_n=remove_c):
    df = df.drop(column_n,axis=1)
    return df


# Impute null values with None
# 用None来赋空值
def null_process(feature_df):
    for col in text_f:
        feature_df.loc[feature_df[col].isnull(), col] = "None"
    return feature_df
 
def clean_dataset(df):
    # remove unused column
    # 移除未使用的列
    df = remove_unused_c(df)
    #impute null values
    # 输入空值
    df = null_process(df)
    return df
 
# Cleaning text from unused characters
# 从未使用的字符中清除文本
def clean_text(text):
    text = str(text).replace(r'http[\w:/\.]+', ' ')  # removing urls删除网址
    text = str(text).replace(r'[^\.\w\s]', ' ')  # remove everything but characters and punctuation 删除除字符和标点符号以外的所有内容
    text = str(text).replace('[â-zA-Z]', ' ')
    text = str(text).replace(r'\s\s+', ' ')
    text = text.lower().strip()
    #text = ' '.join(text)   
    return text
 
# Nltk Preprocessing include:  Nltk预处理包括:
# Stop words, Stemming and Lemmetization 停止单词，词干和词根化
# For our project we use only Stop word removal 对于我们的项目，我们只使用停止字删除
def nltk_preprocess(text):
    text = clean_text(text)
    wordlist = re.sub(r'[^\w\s]', '', text).split()
    #text = ' '.join([word for word in wordlist if word not in stopwords_dict])单词列表中的单词对单词如果单词不在stopwords_dict中
    #text = [ps.stem(word) for word in wordlist if not word in stopwords_dict]单词列表中单词的ps.stem（单词）如果不是stopwords_dict中的单词
    text = ' '.join([wnl.lemmatize(word) for word in wordlist if word not in stopwords_dict])
    return  text

# Perform data cleaning on train and test dataset by calling clean_dataset function
# 通过调用clean_dataset函数对训练数据集和测试数据集进行数据清洗
df = clean_dataset(news_d)
# apply preprocessing on text through apply method by calling the function nltk_preprocess
# 通过Apply方法调用函数nltk_preprocess对文本应用预处理
df["text"] = df.text.apply(nltk_preprocess)
# apply preprocessing on title through apply method by calling the function nltk_preprocess
# 通过Apply方法调用函数nltk_preprocess对标题应用预处理
df["title"] = df.title.apply(nltk_preprocess)
 
# Dataset after cleaning and preprocessing step
# 数据集清洗和预处理步骤
df.head()


""" 
#所有单词的词云
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
 
# initialize the word cloud 初始化词云
wordcloud = WordCloud( background_color='black', width=800,
height=600)
# generate the word cloud by passing the corpus 通过传递语料库生成词云
text_cloud = wordcloud.generate(' '.join(df['text']))
# plotting the word cloud 绘制单词云
plt.figure(figsize=(20,30))
plt.imshow(text_cloud)
plt.axis('off')
plt.show()

#可靠新闻的词云
true_n = ' '.join(df[df['label']==0]['text'])
wc = wordcloud.generate(true_n)
plt.figure(figsize=(20,30))
plt.imshow(wc)
plt.axis('off')
plt.show()

#假新闻的词云
fake_n = ' '.join(df[df['label']==1]['text'])
wc= wordcloud.generate(fake_n)
plt.figure(figsize=(20,30))
plt.imshow(wc)
plt.axis('off')
plt.show() 


 #N-grams
def plot_top_ngrams(corpus, title, ylabel, xlabel="Number of Occurences", n=2):
   #Utility function to plot top n-grams绘制顶部n-grams的效用函数
  true_b = (pd.Series(nltk.ngrams(corpus.split(), n)).value_counts())[:20]
  true_b.sort_values().plot.barh(color='blue', width=.9, figsize=(12, 8))
  plt.title(title)
  plt.ylabel(ylabel)
  plt.xlabel(xlabel)
  plt.show()
#2-gram (bigram)
#可靠新闻bigram 20大经常发生的真实新闻八卦  (不断重复这张图)
plot_top_ngrams(true_n, 'Top 20 Frequently Occuring True news Bigrams', "Bigram", n=2)

#假新闻bigram
plot_top_ngrams(fake_n, 'Top 20 Frequently Occuring Fake news Bigrams', "Bigram", n=2)

#3-gram（trigram）
#可靠新闻trigram
plot_top_ngrams(true_n, 'Top 20 Frequently Occuring True news Trigrams', "Trigrams", n=3)

#假新闻Trigram
plot_top_ngrams(fake_n, 'Top 20 Frequently Occuring Fake news Trigrams', "Trigrams", n=3)
 """


import torch
from transformers.file_utils import is_tf_available, is_torch_available, is_torch_tpu_available
from transformers import BertTokenizerFast, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import numpy as np
from sklearn.model_selection import train_test_split
 
import random

#我们希望即使重新启动环境也能使结果重现:
def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in ``random``,
``numpy``, ``torch`ànd/or ``tf`` (if
    installed).
 
    Args:
        seed (:obj:ìnt`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    if is_torch_available():
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # ^^ safe to call this function even if cuda is not available
    if is_tf_available():
        import tensorflow as tf
 
        tf.random.set_seed(seed)
 
set_seed(1)

# the model we gonna train, base uncased BERT
# check text classification models here: https://huggingface.co/models?filter=text-classification
model_name = "bert-base-uncased"
# max sequence length for each document/sentence sample 每个文档/句子样本的最大序列长度
max_length = 512

# load the tokenizer 加载标记器
tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

#从文本、作者和标题列中清除NaN值:
news_df = news_d[news_d['text'].notna()]
news_df = news_df[news_df["author"].notna()]
news_df = news_df[news_df["title"].notna()]

#创建一个函数，将数据集作为Pandas数据框架，并将文本和标签的训练/验证分割作为列表返回:
def prepare_data(df, test_size=0.2, include_title=True, include_author=True):
  texts = []
  labels = []
  for i in range(len(df)):
    text = df["text"].iloc[i]
    label = df["label"].iloc[i]
    if include_title:
      text = df["title"].iloc[i] + " - " + text
    if include_author:
      text = df["author"].iloc[i] + " : " + text
    if text and label in [0, 1]:
      texts.append(text)
      labels.append(label)
  return train_test_split(texts, labels, test_size=test_size)
 
train_texts, valid_texts, train_labels, valid_labels = prepare_data(news_df)

#确保标签和文本具有相同的长度:
print(len(train_texts), len(train_labels))
print(len(valid_texts), len(valid_labels))

#使用BERT标记器标记数据集，
# 当传递' max_length '时截断数据集，
# 当小于' max_length '时用0填充
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
valid_encodings = tokenizer(valid_texts, truncation=True, padding=True, max_length=max_length)

#将编码转换为PyTorch数据集:
class NewsGroupsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
 
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor([self.labels[idx]])
        return item
 
    def __len__(self):
        return len(self.labels)
 
# convert our tokenized data into a torch Dataset 将我们的标记数据转换为torch数据集
train_dataset = NewsGroupsDataset(train_encodings, train_labels)
valid_dataset = NewsGroupsDataset(valid_encodings, valid_labels)

#load our BERT transformer model
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# num_labels设置为2，因为它是二进制分类。
# 下面的函数是一个回调函数，用于计算每个验证步骤的准确性:
from sklearn.metrics import accuracy_score
 
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function 使用sklearn函数计算精度
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

#让我们初始化训练参数:
training_args = TrainingArguments(
    output_dir='./results',          # output directory 输出目录
    num_train_epochs=1,              # total number of training epochs 训练周期的总数
    per_device_train_batch_size=10,  # batch size per device during training 10 培训期间每个设备的批量大小
    per_device_eval_batch_size=20,   # batch size for evaluation 20 评估批次大小
    warmup_steps=100,                # number of warmup steps for learning rate scheduler 学习率调度器的热身步骤数
    logging_dir='./logs',            # directory for storing logs 日志存放路径
    load_best_model_at_end=True,     # load the best model when finished training (default metric is loss) 训练结束后加载最佳模型(默认指标为损失)
    # but you can specify `metric_for_best_model' argument to change to accuracy or other metric 可以指定' metric_for_best_model'以更改精度或其他度量
    logging_steps=200,               # log & save weights each logging_steps 记录并保存每个logging_steps的权重
    save_steps=200,
    evaluation_strategy="steps",     # evaluate each `logging_steps` 计算每个“logging_steps”日志记录步骤
)
 
#实例化训练器
trainer = Trainer(
    model=model,                         # the instantiated Transformers model to be trained 要训练的实例化transformer模型
    args=training_args,                  # training arguments, defined above 上面定义的训练参数
    train_dataset=train_dataset,         # training dataset 训练数据集
    eval_dataset=valid_dataset,          # evaluation dataset 评估数据集
    compute_metrics=compute_metrics,     # the callback that computes metrics of interest 计算感兴趣指标的回调
)
 
torch.cuda.empty_cache()
# train the model
trainer.train()


#训练后评估当前模型
trainer.evaluate()

# saving the fine tuned model & tokenizer 保存微调模型和标记器
model_path = "fake-news-bert-base-uncased"
model.save_pretrained(model_path)
tokenizer.save_pretrained(model_path)

#创建一个函数，接受article文本作为参数，并返回它是否是假的:
def get_prediction(text, convert_to_label=False):
    # prepare our text into tokenized sequence 准备我们的文本为标记序列
    inputs = tokenizer(text, padding=True, truncation=True,
max_length=max_length, return_tensors="pt").to("cuda")
    # perform inference to our model 对我们的模型进行推理
    outputs = model(**inputs)
    # get output probabilities by doing softmax 通过softmax得到输出概率
    probs = outputs[0].softmax(1)
    # executing argmax function to get the candidate label 执行argmax函数获取候选标签
    d = {
        0: "reliable",
        1: "fake"
    }
    if convert_to_label:
      return d[int(probs.argmax())]
    else:
      return int(probs.argmax())


real_news = """
Tim Tebow Will Attempt Another Comeback, This Time in Baseball - The New York Times",Daniel
Victor,"If at first you don’t succeed, try a different sport. Tim Tebow, who was a Heisman  
quarterback at the University of Florida but was unable to hold an N. F. L. job, is pursuing a career in
Major League Baseball. He will hold a workout for M. L. B. teams this month, his agents told ESPN
and other news outlets. “This may sound like a publicity stunt, but nothing could be further from the
truth,” said Brodie Van Wagenen,   of CAA Baseball, part of the sports agency CAA Sports, in the
statement. “I have seen Tim’s workouts, and people inside and outside the industry  —   scouts,
executives, players and fans  —   will be impressed by his talent. ” It’s been over a decade since
Tebow, 28, has played baseball full time, which means a comeback would be no easy task. But the
former major league catcher Chad Moeller, who said in the statement that he had been training Tebow
in Arizona, said he was “beyond impressed with Tim’s athleticism and swing. ” “I see bat speed and
power and real baseball talent,” Moeller said. “I truly believe Tim has the skill set and potential to
achieve his goal of playing in the major leagues and based on what I have seen over the past two
months, it could happen relatively quickly. ” Or, take it from Gary Sheffield, the former   outfielder.
News of Tebow’s attempted comeback in baseball was greeted with skepticism on Twitter. As a junior
at Nease High in Ponte Vedra, Fla. Tebow drew the attention of major league scouts, batting . 494 with
four home runs as a left fielder. But he ditched the bat and glove in favor of pigskin, leading Florida to
two national championships, in 2007 and 2009. Two former scouts for the Los Angeles Angels told
WEEI, a Boston radio station, that Tebow had been under consideration as a high school junior.
“’x80’x9cWe wanted to draft him, ’x80’x9cbut he never sent back his information card,” said one of
the scouts, Tom Kotchman, referring to a questionnaire the team had sent him. “He had a strong arm
and had a lot of power,” said the other scout, Stephen Hargett. “If he would have been there his senior
year he definitely would have had a good chance to be drafted. ” “It was just easy for him,” Hargett
added. “You thought, If this guy dedicated everything to baseball like he did to football how good
could he be?” Tebow’s high school baseball coach, Greg Mullins, told The Sporting News in 2013 that
he believed Tebow could have made the major leagues. “He was the leader of the team with his
passion, his fire and his energy,” Mullins said. “He loved to play baseball, too. He just had a bigger fire
for football. ” Tebow wouldn’t be the first athlete to switch from the N. F. L. to M. L. B. Bo Jackson
had one   season as a Kansas City Royal, and Deion Sanders played several years for the Atlanta Braves
with mixed success. Though Michael Jordan tried to cross over to baseball from basketball as a    in
1994, he did not fare as well playing one year for a Chicago White Sox minor league team. As a
football player, Tebow was unable to match his college success in the pros. The Denver Broncos
drafted him in the first round of the 2010 N. F. L. Draft, and he quickly developed a reputation for
clutch performances, including a memorable   pass against the Pittsburgh Steelers in the 2011 Wild
Card round. But his stats and his passing form weren’t pretty, and he spent just two years in Denver
before moving to the Jets in 2012, where he spent his last season on an N. F. L. roster. He was cut
during preseason from the New England Patriots in 2013 and from the Philadelphia Eagles in 2015.
"""
 
get_prediction(real_news, convert_to_label=True)
 
# read the test set
test_df = pd.read_csv("test.csv")
 
test_df.head()
 
# make a  of the testing set
new_df = test_df()
 
# add a new column that contains the author, title and article content
new_df["new_text"] = new_df["author"].astype(str) + " : " + new_df["title"].astype(str) + " - " + new_df["text"].astype(str)
new_df.head()
 
# get the prediction of all the test set
new_df["label"] = new_df["new_text"].apply(get_prediction)
 
# make the submission file
final_df = new_df[["id", "label"]]
final_df.to_csv("submit_final.csv", index=False)