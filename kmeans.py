import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
# Optional for pixel colors, but can help
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle  # For sampling pixels if image is too large

# --- 配置参数 ---
INPUT_IMAGE_DIR = 'cloud'  # 包含原始云层图像的文件夹
OUTPUT_IMAGE_DIR = 'output_segmented_clouds'  # 保存分割后图像的文件夹
N_CLUSTERS_PER_IMAGE = 2  # 将每张图像的像素分为多少个簇 (云 vs 背景)
RESIZE_DIM = (200, 200)  # 可选：在处理前调整图像大小以加快速度，None则不调整
MAX_PIXELS_FOR_KMEANS = 100000  # 可选：如果图像像素过多，随机采样一部分像素进行KMeans训练，以提高效率


def segment_image_with_kmeans(image_path, n_clusters, resize_dim=None, max_pixels=None):
    """
    使用K-Means对单个图像的像素进行聚类分割。
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"警告：无法读取图像 {image_path}，跳过。")
            return None

        # 可选：调整图像大小
        if resize_dim:
            img_display = cv2.resize(
                img, resize_dim, interpolation=cv2.INTER_AREA)
            img_for_kmeans = img_display.copy()
        else:
            img_display = img.copy()  # 用于最后生成结果
            img_for_kmeans = img.copy()

        # 将图像从 BGR 转换为 RGB (KMeans通常在RGB上表现更好，且与我们视觉感知一致)
        img_rgb = cv2.cvtColor(img_for_kmeans, cv2.COLOR_BGR2RGB)

        # 将图像数据展平成像素列表 (N_pixels, 3)
        pixel_values = img_rgb.reshape((-1, 3))

        # 可选：如果像素过多，进行采样
        if max_pixels and pixel_values.shape[0] > max_pixels:
            print(
                f"  图像 {os.path.basename(image_path)} 像素过多 ({pixel_values.shape[0]}), 采样 {max_pixels} 个像素进行KMeans训练。")
            pixel_values_for_fit = shuffle(
                pixel_values, random_state=42, n_samples=max_pixels)
        else:
            pixel_values_for_fit = pixel_values

        # 转换为 float 类型，KMeans 需要
        pixel_values_float = np.float32(pixel_values_for_fit)
        all_pixel_values_float = np.float32(pixel_values)

        # K-Means 聚类
        # print(f"  对 {os.path.basename(image_path)} 进行K-Means像素聚类 (K={n_clusters})...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        kmeans.fit(pixel_values_float)

        # 获取每个像素的标签和聚类中心
        labels = kmeans.predict(all_pixel_values_float)  # 在所有像素上预测
        centers = kmeans.cluster_centers_.astype(np.uint8)  # 转换回 uint8 以便显示

        # 根据标签用聚类中心的颜色重新构建图像
        segmented_image_rgb = centers[labels.flatten()]
        segmented_image_rgb = segmented_image_rgb.reshape(img_rgb.shape)

        # 将分割后的图像从 RGB 转回 BGR 以便 OpenCV 保存
        segmented_image_bgr = cv2.cvtColor(
            segmented_image_rgb, cv2.COLOR_RGB2BGR)

        # 如果调整过大小，将分割结果调整回原始（或显示用）的尺寸
        if resize_dim:
            # 如果原始图像也调整了大小用于显示，则分割图也应是这个大小
            final_segmented_image = segmented_image_bgr  # 已经是resize_dim了
        else:
            # 如果没有resize_dim，img_rgb就是原始尺寸，segmented_image_rgb也是原始尺寸
            final_segmented_image = segmented_image_bgr

        # --- 尝试创建二值化蒙版 (云 vs 背景) ---
        # 假设云通常是较亮的簇
        # 计算每个簇中心的平均亮度 (简单方法：平均RGB值)
        center_brightness = np.mean(centers, axis=1)

        # 找到最亮的簇的索引 (假设为云)
        # 注意：这个假设可能不总是成立，取决于天空和云的具体颜色
        cloud_cluster_label = np.argmax(center_brightness)

        # 创建一个二值蒙版
        # 云像素为白色 (255), 背景像素为黑色 (0)
        binary_mask = np.zeros(labels.shape, dtype=np.uint8)
        binary_mask[labels == cloud_cluster_label] = 255
        binary_mask = binary_mask.reshape(
            img_rgb.shape[:2])  # Reshape to 2D mask

        # 将二值蒙版转换为3通道图像以便与分割图并排显示或单独保存
        binary_mask_3channel = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)

        # 可以选择返回哪个图像：
        # return final_segmented_image # 返回用簇中心颜色重绘的图像
        return binary_mask_3channel  # 返回二值化蒙版

    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {e}")
        return None


def main():
    if not os.path.exists(INPUT_IMAGE_DIR):
        print(f"错误：输入文件夹 '{INPUT_IMAGE_DIR}' 不存在。请创建并放入图像。")
        return

    if not os.path.exists(OUTPUT_IMAGE_DIR):
        os.makedirs(OUTPUT_IMAGE_DIR)
        print(f"创建输出文件夹: {OUTPUT_IMAGE_DIR}")

    supported_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(
        INPUT_IMAGE_DIR) if f.lower().endswith(supported_extensions)]

    if not image_files:
        print(f"错误：输入文件夹 '{INPUT_IMAGE_DIR}' 中没有找到支持的图像文件。")
        return

    print(f"找到 {len(image_files)} 张图像进行处理...")

    for filename in image_files:
        input_path = os.path.join(INPUT_IMAGE_DIR, filename)
        output_path = os.path.join(OUTPUT_IMAGE_DIR, f"segmented_{filename}")

        print(f"正在处理: {filename}")
        segmented_result = segment_image_with_kmeans(
            input_path, N_CLUSTERS_PER_IMAGE, resize_dim=RESIZE_DIM, max_pixels=MAX_PIXELS_FOR_KMEANS)

        if segmented_result is not None:
            cv2.imwrite(output_path, segmented_result)
            print(f"  已保存分割结果到: {output_path}")
        else:
            print(f"  未能处理或分割图像: {filename}")

    print("\n所有图像处理完成。")


if __name__ == '__main__':
    main()
