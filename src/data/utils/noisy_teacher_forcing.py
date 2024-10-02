import torch

class NoisyTeacherForcing():
    def __init__(self, vocab_size, noise_prob=0., low=4):
        self.noise_prob = torch.Tensor([noise_prob])
        self.vocab_size = vocab_size
        self.low = low

    def __call__(self, x):
        if self.noise_prob == 0:
            return x
        noise = torch.randint(low=self.low, high=self.vocab_size, size=x.shape)
        prob = torch.rand(size=x.shape)
        prob[:,0] = 1
        if x.is_cuda:
            noise = noise.cuda()
            prob = prob.cuda()
            self.noise_prob = self.noise_prob.cuda()
        return torch.where(prob>self.noise_prob,x,noise)

if __name__ == "__main__":
    NTF = NoisyTeacherForcing(vocab_size=89, noise_prob=0.3)
    x = torch.LongTensor([[2,5,6,7,78,5,6,7,3, 1, 1, 1, 1, 1], [2,5,6,7,78,5,6,7,3,1,1,1,3,1]]).cuda()
    print(x-NTF(x))
