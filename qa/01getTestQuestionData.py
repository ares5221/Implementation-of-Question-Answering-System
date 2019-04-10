#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
import csv
from bertClientRun2 import getBestAnswer2bySimilyQuestionByq2qModel
from bertClient import getBestAnswer

path = os.path.abspath('..')
filePath = path + '/data/qa-clean-data.csv'
save_log_dir = path + '/log/Fun2Log.csv'
save_log_dir1 = path + '/log/Fun1Log.csv'
save_log_dir2 = path + '/log/testLog.csv'
save_log_dir3 = path + '/log/testLog1.csv'
def getTestQ():
    qdata = []
    with open(filePath, 'r', encoding='utf-8') as csvfile:
        read = csv.reader(csvfile)
        for i in read:
            qdata.append(i[0])
    print(len(qdata), '-->', qdata)
    qset = set(qdata)
    print(qset)
    qqlist = list(qset)
    print(len(qqlist), '-->', qqlist)
    return qqlist[0:100]

def getTestQ2():
    qlist = ['注意力不集中怎么办？',
             '如何让孩子喜欢学英语',
             '学生恋爱问题',
             '如何防止孩子抄作业',
             '孩子学习没有热情怎么办',
             '如何让孩子喜欢哲学',
             '孩子喜欢打游戏机怎么办',
             '学生上课总是讲话怎么办',
             '想要调动学习积极性,有什么好的建议吗',
             '老师们怎么能够管理好课堂秩序',
             '对于学生偏科有什么好的建议吗',
             '怎么样可以让家长配合老师完成教学任务',
             '好动淘气的学生怎么办',
             '想要提高孩子的阅读能力',
             '如何树立好的习惯',
             '对于性格内向的孩子,怎么鼓励',
             '孩子不愿意上学怎么办',
             '期末时候如何合理安排孩子的复习计划',
             '上课喜欢睡觉怎么办',
             '女生喜欢化妆,怎么教育',
             '如何合理安排假期作业',
             '如何对待单亲家庭的孩子',
             '如何快速建立班级凝聚力',
             '高中学生沉迷游戏怎么办',
             '家长该秉持怎样的育人理念，才能培养孩子养成良好的学习、生活习惯',
             '孩子不喜欢阅读怎么办',
             '如何与情绪不好的孩子相处',
             '如何帮助班级里的差生',
             '怎么合理处理师生间的矛盾',
             '提高孩子的作文水平',
             '如何提高学习小组的讨论热情'
             ]
    return qlist

if __name__ == '__main__':
    print('开始查询相似问题--->')
    data = getTestQ()
    # for i in range(100):
    #     ss = data[i]
    #     res1, res2 = getBestAnswer2bySimilyQuestionByq2qModel(ss)
    #     news = [ss, res1]
    #     with open(save_log_dir, 'a', newline='', encoding='utf-8') as csvfile:
    #         spamwriter = csv.writer(csvfile)
    #         spamwriter.writerow(news)
    # for i in range(100):
    #     ss = data[i]
    #     res1, res2 = getBestAnswer(ss)
    #     news = [ss, res1]
    #     with open(save_log_dir1, 'a', newline='', encoding='utf-8') as csvfile:
    #         spamwriter = csv.writer(csvfile)
    #         spamwriter.writerow(news)

    testdata = getTestQ2()
    for i in range(len(testdata)):
        ss = testdata[i]
        res1, res2 = getBestAnswer2bySimilyQuestionByq2qModel(ss)
        # res1, res2 = getBestAnswer(ss)
        news = [ss, res1]
        with open(save_log_dir3, 'a', newline='', encoding='utf-8') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(news)
    print('测试问题与相似问题已经保存在log文件中!!!!!')
