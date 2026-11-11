"""학습 영상의 질환 잔차 방향에 긍정−부정 텍스트 차이를 정렬한다.

학습 시작 전 한 번:
    criterion = ResidualAlignmentLoss(z_train, y_train, image_noise)

매 학습 step (tp, tn은 text encoder의 gradient가 살아 있는 출력):
    loss_align = criterion(tp, tn, disease_index)
    loss = loss_clip + 0.1 * loss_align
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

z_train: [N,F], y_train: [N,D] 이진 라벨, image_noise: [F] 노이즈 방향
tp, tn: [K,F], disease_index: [K] long tensor, 값은 y_train의 열 번호.
입력은 같은 device에 둔다. 고정 이미지 encoder와 학습 text encoder를 사용한다.
학습 영상에서 구한 평면과 질환 방향은 buffer로 고정된다.
"""
import torch
from torch import nn
from torch.nn import functional as F


class ResidualAlignmentLoss(nn.Module):
    @torch.no_grad()
    def __init__(self, z_train, y_train, image_noise):
        super().__init__()
        z = F.normalize(z_train.detach().float(), dim=-1)
        y = y_train.detach().to(device=z.device, dtype=z.dtype)
        if z.ndim != 2 or y.ndim != 2 or len(z) != len(y):
            raise ValueError('z_train [N,F], y_train [N,D]의 행 순서를 맞춰주세요.')
        if not torch.all((y == 0) | (y == 1)):
            raise ValueError('y_train에는 양성 1, 음성 0을 입력하세요.')
        count = y.sum(0)
        if torch.any((count == 0) | (count == len(z))):
            raise ValueError('학습 자료의 각 질환에 양성·음성 영상이 모두 필요합니다.')
        b = image_noise.detach().to(device=z.device, dtype=z.dtype)
        if b.shape != (z.shape[1],) or b.norm() < 1e-8:
            raise ValueError('image_noise는 길이가 F인 유효한 노이즈 방향이어야 합니다.')
        b = F.normalize(b, dim=0)
        mean = z.mean(0)
        e = mean - (mean @ b)*b
        if e.norm() < 1e-8:
            raise ValueError('평면을 정의할 독립적인 평균 방향과 노이즈 방향이 필요합니다.')
        q = torch.stack((b, F.normalize(e, dim=0)), dim=1)
        a = y.T @ z/count[:, None] - (1-y).T @ z/(len(z)-count)[:, None]
        a = a-(a @ q) @ q.T
        self.register_buffer('basis', q)
        self.register_buffer('target', F.normalize(a, dim=-1))
        self.register_buffer('valid', a.norm(dim=-1) > 1e-8)

    def forward(self, positive, negative, disease_index):
        # float32 투영: AMP 학습에서도 작은 잔차 방향의 정밀도를 유지한다.
        with torch.autocast(device_type=positive.device.type, enabled=False):
            dt = F.normalize(positive.float(), dim=-1)-F.normalize(negative.float(), dim=-1)
            w = dt-(dt @ self.basis) @ self.basis.T
            similarity = (F.normalize(w, dim=-1)*self.target[disease_index]).sum(-1)
            weight = self.valid[disease_index].to(w.dtype)
            return ((1-similarity)*weight).sum()/weight.sum().clamp_min(1)


if __name__ == '__main__':
    # 외부 데이터 없이 loss의 정방향/역방향 및 gradient 전달을 확인한다.
    z = torch.tensor([[1., 1., 2.], [1., 1., 3.], [1., 1., -2.], [1., 1., -3.]])
    criterion = ResidualAlignmentLoss(z, torch.tensor([[1], [1], [0], [0]]), torch.tensor([1., 0., 0.]))
    p = torch.tensor([[0., 0., 1.]])
    index = torch.tensor([0])
    torch.testing.assert_close(criterion(p, -p, index), torch.tensor(0.))
    torch.testing.assert_close(criterion(-p, p, index), torch.tensor(2.))
    p = torch.tensor([[1., 0., 1.]], requires_grad=True)
    n = torch.tensor([[0., 1., -1.]], requires_grad=True)
    criterion(p, n, index).backward()
    assert torch.isfinite(p.grad).all() and p.grad.norm() > 0
    assert torch.isfinite(n.grad).all() and n.grad.norm() > 0
    print('정렬 loss 검증 통과: 정방향 0, 역방향 2, 양쪽 텍스트 gradient 전달')
