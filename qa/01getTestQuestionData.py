#!/usr/bin/env python
# _*_ coding:utf-8 _*_
import os
import csv
from bertClientRun2 import getBestAnswer2bySimilyQuestionByq2qModel
from bertClient import getBestAnswer
from bertClientRun3 import getBestAnswer3bySimilyQuestionByQ2QandQ2AModel
from bertClientPro import getBestAnswerByKDTree

path = os.path.abspath('..')
filePath = path + '/data/qa-clean-data425.csv'
save_log_dir = path + '/log/Fun2Log.csv'    # 随机选100个问题集中问题来测试方法三
save_log_dir1 = path + '/log/Fun1Log.csv'   # 随机选100个问题集中问题来测试用方法一
save_log_dir2 = path + '/log/testLog.csv'   # 用方法三余弦比较问题相似度获取相似问题后用MLP比较五个答案中合适的答案其对应的问题作为相似问题输出
save_log_dir3 = path + '/log/testLog1.csv'  # 用方法一余弦比较问题相似度获取相似问题
save_log_dir4 = path + '/log/testLog2.csv'  # 用方法三测试，输出问题，获取的相似问题，对应的答案
save_log_dir5 = path + '/log/testLog5.csv'  # 用BallTree，输出问题，获取的相似问题，对应的答案
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
             '如何提高学习小组的讨论热情',
             '学生们上课讨论问题的有效性不高，如何能够提高',
             '上课走神，注意力不集中怎么办'
             '如何让孩子感受到班集体的温暖，热爱班集体',
             '学生出现厌学的状态，不想上学，如何引导',
             '学生在网上查答案，怎么让孩子不在百度作业帮上查答案',
             '在班级管理中如何发挥班干部的作业',
             '对于成绩不理想的考生怎么安抚',
             '如何帮助考试成绩不理想的考生',
             '怎么让孩子喜欢数学',
             '学生上课就喜欢睡觉怎么办',
             '教学生如何控制情绪',
             '有什么好的学英语方法吗或学英语的诀窍',
             '学英语有什么好办法吗',
             '怎么教好一年级的拼音，比如拼读音节',
             '怎么让老师和家长有效沟通，帮助孩子健康成长',
             '如何找到孩子的闪光点',
             '学生不爱写作业怎么办',
             '学生怎么培养好的阅读习惯',
             '学生不交作业怎么办',
             '怎么样能成为一个好老师',
             '如何培养学生的数学思维能力',
             '学生不喜欢上音乐课怎么办',
             '音乐课纪律不好，学生不喜欢上欣赏课怎么办',
             '学生不喜欢写作业，如何让家庭作业有效又有趣呢',
             '如何布置有效又有趣的家庭作业呢',
             '如何调动孩子的学习兴趣',
             '怎么让学生写字的姿态得到教正',
             '如何激发孩子学习的自觉性',
             '如何引导学生学会合理管理时间',
             '如何帮助孩子更好的适应小学生活',
             '如何培养学生认真倾听的习惯',
             '怎么样可以强化学生的心理素质',
             '怎么可以帮助学生更快更准的记住方程式',
             '如何做到赏识教育',
             '培养学生自主管理和自我教育能力的关键是什么',
             '如何处理好老师自己的情绪呢',
             '为什么节假日学生作业比较差',
             '怎样让孩子树立自信，活泼开朗？',
             '家校沟通的策略有哪些？',
             '对于留守儿童怎样开展更好的家校合作呢',
             '作为一名校长是如何评价教师的？如何能够体现教师的发展呢',
             '在帮助问题学生时最关键的是什么？',
             '很多时候学生不懂得如何控制自己的情绪，冲动之下就会发生打架事件。',
             '面对缺乏情绪控制能力的学生，在处理学生因情绪问题发生的言语及身体冲突事件时，有哪些好的策略和办法？',
             '如何分析学生冲突的原因，以及合理处置的方式',
             '样组织一堂有效的一年级数学课',
             '推荐您认为比较适合的亲子互动游戏',
             '如何提升学生自主管理能力？',
             '如何管理好差下生，使之进步？',
             '如果提升一年级的学生自主管理能力？',
             '怎样鼓励和树立有特长却不敢参加活动的孩子？',
             '怎样指导一年级的孩子写好汉字？',
             '班级里总有些学生会违反课堂纪律，挑战校园规则。',
             '有哪些办法去关注学生',
             '如何搞好幼小衔接，让孩子尽快适应小学一年级的学习？',
             '评价一节好课的标准是什么',
             '如何与包办型家长沟通？让他们树立正确的教养方式，能够更好地配合育人工作',
             '评价一个好老师的标准是什么？',
             '农村的家长为什么对孩子的学习不那么关注，需要反复地做工作',
             '教师心理健康的标准是什么？',
             '如何让传统文化走进课堂？',
             '如何看待变相体罚',
             '谈谈学生发展核心素养对我们教师提出了什么新的要求吗',
             '学生不小心磕伤，家长到校兴师问罪，应怎样处理较合适?',
             '如何处理学生偷窃行为？',
             '改掉学生做事拖拖拉垃的坏习惯',
             '班级里的孩子成绩差异大该怎么办',
             '孩子爱写作业的动力培养',
             '怎样让学习成为孩子的一种习惯',
             '如何提高学生汉语口头表达能力呢',
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
        # res1, res2 = getBestAnswer2bySimilyQuestionByq2qModel(ss)
        # res1, res2 = getBestAnswer(ss)
        res1, res2 = getBestAnswerByKDTree(ss)
        # res1, res2, res3 = getBestAnswer3bySimilyQuestionByQ2QandQ2AModel(ss)
        news = [ss, res1, res2]   # 问题，获取的相似问题，mlp后合适的答案， 合适答案对应的问题
        with open(save_log_dir5, 'a', newline='', encoding='utf-8') as csvfile:
            spamwriter = csv.writer(csvfile)
            spamwriter.writerow(news)
    print('测试问题与相似问题已经保存在log文件中!!!!!')
