import os
import re
import pandas as pd
import praw
from bertopic import BERTopic
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt

# --- 1. SETUP: Reddit API & Subreddits ---
print("Connecting to Reddit...")
reddit = praw.Reddit(
    client_id=os.environ['REDDIT_CLIENT_ID'],
    client_secret=os.environ['REDDIT_CLIENT_SECRET'],
    user_agent="reddit-trends-mvp by u/your_reddit_username", # 最好换成你自己的Reddit用户名
)

subreddits_to_scan = ['futurology', 'technology', 'singularity']
posts_data = []
docs = []

# --- 2. DATA COLLECTION ---
print(f"Fetching posts from: {subreddits_to_scan}...")
for sub_name in subreddits_to_scan:
    subreddit = reddit.subreddit(sub_name)
    # 抓取过去24小时的热门帖子，每个版块最多50篇
    for post in subreddit.top(time_filter="day", limit=50):
        if not post.stickied: # 跳过置顶帖
            full_text = post.title + " " + post.selftext
            posts_data.append({'title': post.title, 'text': post.selftext, 'full_text': full_text})

if not posts_data:
    print("No posts found. Exiting.")
    exit()

docs = [d['full_text'] for d in posts_data]
print(f"Collected {len(docs)} documents.")

# --- 3. DATA CLEANING (Basic) ---
print("Cleaning documents...")
def clean_text(text):
    text = text.lower()  # 转小写
    text = re.sub(r'http\S+', '', text)  # 去除URL
    text = re.sub(r'\[.*?\]', '', text)  # 去除Markdown链接等
    text = re.sub(r'[^a-z\s]', '', text) # 只保留字母和空格
    return text

cleaned_docs = [clean_text(doc) for doc in docs]

# --- 4. TOPIC MODELING ---
print("Performing topic modeling with BERTopic...")
topic_model = BERTopic(verbose=False, min_topic_size=3) # 至少3个文档才能形成一个话题
topics, _ = topic_model.fit_transform(cleaned_docs)

# --- 5. SENTIMENT ANALYSIS ---
print("Performing sentiment analysis...")
analyzer = SentimentIntensityAnalyzer()
sentiments = [analyzer.polarity_scores(doc)['compound'] for doc in docs]

# 结果整合
df = pd.DataFrame({
    'doc': docs,
    'topic': topics,
    'sentiment': sentiments
})
# 计算每个话题的平均情绪
topic_sentiment = df.groupby('topic')['sentiment'].mean().reset_index()

# --- 6. VISUALIZATION ---
print("Generating visualization...")
# 获取Top 10话题的信息 (排除离群点-1)
top_topics_info = topic_model.get_topic_info()
if top_topics_info.iloc[0]['Topic'] == -1:
    top_topics_info = top_topics_info.iloc[1:11]
else:
    top_topics_info = top_topics_info.iloc[0:10]

# 创建条形图
plt.figure(figsize=(12, 8))
plt.bar(top_topics_info['Name'], top_topics_info['Count'], color='skyblue')
plt.title('Top 10 Topics of the Day', fontsize=16)
plt.ylabel('Number of Posts', fontsize=12)
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.tight_layout() # 调整布局防止标签重叠

# 确保输出目录存在
os.makedirs('output', exist_ok=True)
chart_path = 'output/topic_chart.png'
plt.savefig(chart_path)
print(f"Chart saved to {chart_path}")

# --- 7. GENERATE MARKDOWN REPORT ---
print("Generating Markdown report...")
report_path = 'output/report.md'
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("# Reddit 每日趋势洞察报告\n\n")
    f.write(f"**报告生成时间:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n\n")
    f.write("## 今日热门话题 Top 10\n\n")
    f.write("![Top 10 Topics](./topic_chart.png)\n\n") # 嵌入图片（注意相对路径）
    f.write("## 话题详情与情绪分析\n\n")
    f.write("| 话题关键词 | 帖子数量 | 平均情绪 (-1负 ~ 1正) |\n")
    f.write("|---|:---:|:---:|\n")
    
    for _, row in top_topics_info.iterrows():
        topic_num = row['Topic']
        sentiment_score = topic_sentiment.query(f"topic == {topic_num}")['sentiment'].values[0]
        f.write(f"| `{row['Name']}` | {row['Count']} | **{sentiment_score:.2f}** |\n")

print(f"Report saved to {report_path}")
print("All tasks completed successfully!")
