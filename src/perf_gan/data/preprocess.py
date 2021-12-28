import torch
import torch.nn as nn


class Identity(nn.Module):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X

    def inverse_transform(self, X, y=None):
        return X


class PitchTransform(nn.Module):
    def __init__(self, feature_range=(0, 1)):
        self.m = None
        self.M = None
        self.new_m, self.new_M = feature_range

    def fit(self, X):
        self.m = torch.min(X)
        self.M = torch.max(X)
        return self

    def transform(self, X, y=None):
        return ((X - self.m) /
                (self.M - self.m)) * (self.new_M - self.new_m) + self.new_m

    def inverse_transform(self, X, y=None):
        return ((X - self.new_m) /
                (self.new_M - self.new_m)) * (self.M - self.m) + self.m


class LoudnessTransform(nn.Module):
    def __init__(self, feature_range=(0, 1)):
        self.m = None
        self.M = None
        self.new_m, self.new_M = feature_range

    def fit(self, X):
        self.m = torch.min(X)
        self.M = torch.max(X)
        print(self.m)
        print(self.M)
        return self

    def transform(self, X, y=None):
        return ((X - self.m) /
                (self.M - self.m)) * (self.new_M - self.new_m) + self.new_m

    def inverse_transform(self, X, y=None):
        return ((X - self.new_m) /
                (self.new_M - self.new_m)) * (self.M - self.m) + self.m
