import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import numpy as np
from rembg.bg import remove
import io
import os
from decimal import Decimal, ROUND_HALF_UP

def stylechange(input_path, style_path):
    print("input_img_path: ",input_path)
    print("style_img_path: ",style_path)
    # 背景画像のパス
    current_dir = os.path.dirname(os.path.abspath(__file__))
    greenback_path = os.path.join(current_dir, 'img_e_g', 'greenback.jpeg')

    print(greenback_path)
    
    # 画像ファイルをバイトストリームとして読み込み、背景を削除
    with open(input_path, 'rb') as f:
        image_data = f.read()
    result = remove(image_data, alpha_matting=True, alpha_matting_foreground_threshold=240,
                    alpha_matting_background_threshold=10, alpha_matting_erode_structure_size=6)
    cut_img = Image.open(io.BytesIO(result)).convert("RGBA")

    # 画像のアスペクト比を保持しつつリサイズ
    def scale_to_width(img, width):
        height = round(img.height * width / img.width)
        return img.resize((width, height))

    # 背景画像を読み込み、サイズを調整
    bg = Image.open(greenback_path).convert('RGBA').resize(cut_img.size)

    # 画像合成
    merge = Image.alpha_composite(bg, cut_img)

    # VGG19モデルをロードし、パラメータを固定
    vgg = models.vgg19(pretrained=True).features
    for param in vgg.parameters():
        param.requires_grad_(False)
    
    # GPUが利用可能かどうかをチェック
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vgg.to(device)
    
    # 画像を読み込み、前処理を行う関数
    def load_image(image, max_size=400, shape=None):
        if max(image.size) > max_size:
            size = max_size
        else:
            size = max(image.size)
        if shape:
            size = shape
        in_transform = transforms.Compose([
                            transforms.Resize(size),
                            transforms.ToTensor()])
        image = in_transform(image).unsqueeze(0)
        return image
    
    # コンテンツ画像とスタイル画像を読み込み
    content = load_image(merge.convert('RGB')).to(device)
    style = load_image(Image.open(style_path).convert('RGB'), shape=content.shape[-2:]).to(device)

    # Tensorを画像に変換する関数
    def im_convert(tensor):
        image = tensor.to("cpu").clone().detach()
        image = image.numpy().squeeze()
        image = image.transpose(1,2,0)
        image = image.clip(0, 1)
        return image
    
    # 特徴抽出関数
    def get_features(image, model):
        layers = {'0': 'conv1_1', '5': 'conv2_1', '10': 'conv3_1', '19': 'conv4_1', '21': 'conv4_2', '28': 'conv5_1'}
        features = {}
        for name, layer in model._modules.items():
            image = layer(image)
            if name in layers:
                features[layers[name]] = image
        return features
    
    # 特徴量とスタイル画像からグラム行列を計算
    content_features = get_features(content, vgg)
    style_features = get_features(style, vgg)
    def gram_matrix(tensor):
        _, d, h, w = tensor.size()
        tensor = tensor.view(d, h*w)
        gram = torch.mm(tensor, tensor.t())
        return gram
    
    style_grams = {layer : gram_matrix(style_features[layer]) for layer in style_features}
    
    # スタイル損失の重み
    # この部分は以前のコードには含まれていませんでしたが、スタイル転送の重要な部分です。
    style_weights = {'conv1_1': 1., 'conv2_1': 0.75, 'conv3_1': 0.2, 'conv4_1': 0.2, 'conv5_1': 0.2}
    
    # コンテンツとスタイルの損失の重み
    content_weight = 1e3
    style_weight = 1e2
    
    # ターゲット画像をコンテンツ画像からクローンして初期化
    target = content.clone().requires_grad_(True).to(device)
    
    # 最適化器の設定
    optimizer = optim.Adam([target], lr=0.003)
    
    # スタイル転送のプロセスを実行
    steps = 3000  # スタイル転送のイテレーション回数
    print('Generating...')
    for ii in range(1, steps+1):
        target_features = get_features(target, vgg)
        content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
        
        style_loss = 0
        for layer in style_weights:
            target_feature = target_features[layer]
            target_gram = gram_matrix(target_feature)
            style_gram = style_grams[layer]
            layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
            _, d, h, w = target_feature.shape
            style_loss += layer_style_loss / (d * h * w)
        
        total_loss = content_weight * content_loss + style_weight * style_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 一定のステップごとに進捗を表示
        if ii % 50 == 0:
            #print('Step {}: Style Loss: {:(4f}, Content Loss: {:4f}'.format(ii, style_loss.item(), content_loss.item()))
            print('Step:{}/{}({}%)'.format(ii, steps, Decimal(str((ii/steps)*100)).quantize(Decimal('0'), ROUND_HALF_UP)))
    
    # 最終的なターゲット画像を表示
    cutting_result_changed = remove((im_convert(target)* 255).astype(np.uint8),
                alpha_matting=True,
                alpha_matting_foreground_threshold=240,
                alpha_matting_background_threshold=10,
                alpha_matting_erode_structure_size=6)
    style_changed = Image.open(input_path).convert('RGBA')
    cut_img_changed = Image.fromarray(cutting_result_changed).convert('RGBA').resize(style_changed.size)
    merge_changed = Image.alpha_composite(style_changed, cut_img_changed)
    fig, ax = plt.subplots(1, 3, figsize=(20, 10))
    imgs = [style_changed, im_convert(style) , merge_changed]
    names = ['Input', 'Style', 'Output']

    for i, im, name in zip(range(len(ax)), imgs, names):
        img = im
        ax[i].imshow(img)
        ax[i].set_title(name)
        ax[i].axis('off')

    plt.show()
