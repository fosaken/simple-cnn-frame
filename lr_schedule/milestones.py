# -*-coding:utf-8-*-
#
#    @header milestones.py
#    @author  CaoZhihui
#    @date    2020/9/23
#    @abstract:
#


class Milestones(object):

    def __init__(self, optimizer, base_lr, gamma=0.1, milestones=(9999, )):
        self._base_lr = base_lr
        self._gamma = gamma
        self._milestones = milestones

    def update_lr(self, epoch):
        power = 0
        for milestone in self._milestones:
            if epoch > milestone:
                power += 1
            else:
                break



