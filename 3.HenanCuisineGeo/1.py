import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import os
import matplotlib as mpl
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ---------------------- 全局配置（精准调参） ----------------------
mpl.rcParams.update({
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'font.sans-serif': ['SimHei', 'Microsoft YaHei'],
    'axes.unicode_minus': False,
    'text.antialiased': True,
    'lines.antialiased': True
})

# ---------------------- 核心布局配置 ----------------------
CONFIG = {
    # 整体画布
    'main_map_size': (18, 14),    
    'legend_width': 5,            
    # 字体大小（鹤壁市单独缩小）
    'city_font': 12,              
    'food_font': 10,              
    'hebi_city_font': 10,         # 鹤壁市字体缩小
    'hebi_food_font': 8,          # 鹤壁美食字体缩小
    'legend_title_font': 14,      
    'legend_text_font': 11,       
    'zhangliang_font': 13,        
    # 图片与间距
    'img_zoom': 0.28,             
    'img_unified_size': (250, 180),
    'zhangliang_ms': 50,          
    'map_text_gap': 0.04,         
    'legend_row_gap': 0.22,       
    'legend_col_gap': 0.28,       
    # 色彩
    'color_bg': '#ffffff',        
    'color_border': '#00264d',    
    'color_city_text': '#00264d', 
    'color_food_text': '#0059b3', 
    'color_zhangliang': '#d92b2b' 
}

# ---------------------- 城市位置微调参数（核心修复） ----------------------
# 针对指定城市的位置/字体微调
CITY_ADJUST = {
    '三门峡市': {'lon_offset': -0.3, 'lat_offset': 0},  # 向左移
    '安阳市': {'lon_offset': 0, 'lat_offset': 0.15},     # 向上移
    '鹤壁市': {'lon_offset': 0, 'lat_offset': 0},        # 字体缩小（单独配置）
    '濮阳市': {'lon_offset': -0.2, 'lat_offset': 0},     # 向左移
    '济源示范区': {'food': '土馍'}                       # 补充特产
}

# ---------------------- 0. 切换目录 ----------------------
data_dir = "张家诚"
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"❌ 未找到数据目录 '{data_dir}'，请确认文件夹存在。")
os.chdir(data_dir)
print(f"✅ 已切换工作目录到：{os.getcwd()}")

# ---------------------- 1. 加载地理数据 ----------------------
city_geojson_dir = "."
henan_cities = []
print("正在扫描目录中的GeoJSON文件...")
for filename in os.listdir(city_geojson_dir):
    if filename.endswith("_市.geojson") and os.path.isfile(os.path.join(city_geojson_dir, filename)):
        print(f"✅ 找到文件：{filename}")
        henan_cities.append(gpd.read_file(os.path.join(city_geojson_dir, filename)))

if not henan_cities:
    raise ValueError("❌ 未找到任何以 '_市.geojson' 结尾的文件，请检查文件命名是否正确。")
henan_cities = gpd.GeoDataFrame(pd.concat(henan_cities, ignore_index=True)).to_crs("EPSG:4326")

try:
    henan_province = gpd.read_file("河南省_省.geojson").to_crs("EPSG:4326")
except FileNotFoundError:
    raise FileNotFoundError("❌ 未找到 '河南省_省.geojson' 文件，请确认文件存在。")

# ---------------------- 2. 核心数据（补充济源特产） ----------------------
city_food_data = pd.DataFrame({
    'city': [
        '郑州市', '开封市', '洛阳市', '平顶山市', '安阳市', '鹤壁市',
        '新乡市', '焦作市', '濮阳市', '许昌市', '漯河市', '三门峡市',
        '南阳市', '商丘市', '信阳市', '周口市', '驻马店市', '济源示范区'
    ],
    'food': [
        '烩面', '灌汤包', '水席', '饸饹面', '扁粉菜', '缠丝鸭蛋',
        '红焖羊肉', '铁棍山药', '壮馍', '热干面', '胡辣汤', '灵宝肉夹馍',
        '板面', '水煎包', '南湾鱼', '逍遥镇胡辣汤', '正阳花生', '土馍'  # 补充济源特产
    ],
    'food_img': [
        '烩面.jpg', '灌汤包.jpg', '水席.jpg', '饸饹面.jpg', '扁粉菜.jpg', '缠丝鸭蛋.jpg',
        '红焖羊肉.jpg', '铁棍山药.jpg', '壮馍.jpg', '热干面.jpg', '胡辣汤.jpg', '灵宝肉夹馍.jpg',
        '板面.jpg', '水煎包.jpg', '南湾鱼.jpg', '逍遥镇胡辣汤.jpg', '正阳花生.jpg', '土馍.jpg'
    ],
    'lon': [113.65, 114.35, 112.45, 113.28, 114.35, 114.28,
            113.85, 113.28, 115.00, 113.80, 114.02, 111.15,
            112.53, 115.65, 114.08, 114.60, 114.00, 112.58],
    'lat': [34.76, 34.79, 34.62, 33.76, 36.10, 35.75,
            35.30, 35.24, 35.77, 34.02, 33.57, 34.76,
            33.00, 34.44, 32.12, 33.65, 32.98, 35.08]
})

# 张良镇数据
zhangliang = pd.DataFrame({'name': ['张良镇'], 'lon': [112.7058], 'lat': [33.6042]})

# ---------------------- 3. 辅助函数 ----------------------
def load_and_unify_img(img_name, zoom):
    path = os.path.join(".", img_name)
    if os.path.exists(path):
        img_pil = Image.open(path)
        img_pil.thumbnail(CONFIG['img_unified_size'], Image.Resampling.LANCZOS)
        return OffsetImage(img_pil, zoom=zoom, alpha=0.98)
    return None

# ---------------------- 4. 创建画布 ----------------------
# 方案1：保留原布局（精准修复文字），如需环绕布局可切换方案2
fig, (ax_map, ax_legend) = plt.subplots(
    ncols=2,
    figsize=(CONFIG['main_map_size'][0] + CONFIG['legend_width'], CONFIG['main_map_size'][1]),
    dpi=600,
    gridspec_kw={
        'width_ratios': [CONFIG['main_map_size'][0], CONFIG['legend_width']],
        'wspace': 0.02
    }
)
fig.patch.set_facecolor(CONFIG['color_bg'])

# ---------------------- 5. 绘制地图（核心：精准调整文字位置） ----------------------
# 1. 地级市底色
henan_cities.plot(
    ax=ax_map, 
    facecolor='#f0f8ff', 
    edgecolor='#cce5ff', 
    linewidth=1.2, 
    zorder=2,
    alpha=0.9
)
# 2. 省级轮廓
henan_province.plot(
    ax=ax_map, 
    facecolor='none', 
    edgecolor=CONFIG['color_border'], 
    linewidth=3.5, 
    zorder=4
)

# 3. 标注市名和美食名（精准调整）
for idx, row in henan_cities.iterrows():
    centroid = row.geometry.centroid
    city_name = row['city_name'] if 'city_name' in row else row['name']
    
    # 应用位置微调
    lon = centroid.x + CITY_ADJUST.get(city_name, {}).get('lon_offset', 0)
    lat = centroid.y + CITY_ADJUST.get(city_name, {}).get('lat_offset', 0)
    
    # 应用字体大小微调（鹤壁市单独缩小）
    city_font_size = CONFIG['hebi_city_font'] if city_name == '鹤壁市' else CONFIG['city_font']
    food_font_size = CONFIG['hebi_food_font'] if city_name == '鹤壁市' else CONFIG['food_font']
    
    # 市名
    ax_map.text(
        lon, lat + CONFIG['map_text_gap'], 
        city_name, 
        fontsize=city_font_size,
        ha='center', va='bottom', 
        fontweight='bold', 
        color=CONFIG['color_city_text'], 
        zorder=5,
        bbox=dict(boxstyle="round,pad=0.03", facecolor='white', alpha=0.7, edgecolor='none')
    )
    
    # 美食名
    food_row = city_food_data[city_food_data['city'] == city_name]
    if not food_row.empty:
        food_name = food_row.iloc[0]['food']
        ax_map.text(
            lon, lat - CONFIG['map_text_gap'], 
            f'({food_name})', 
            fontsize=food_font_size,
            ha='center', va='top', 
            color=CONFIG['color_food_text'], 
            zorder=5
        )

# 4. 标注张良镇
ax_map.scatter(
    zhangliang.lon, zhangliang.lat, 
    s=CONFIG['zhangliang_ms'], 
    c=CONFIG['color_zhangliang'], 
    edgecolor='white', 
    linewidth=2, 
    zorder=6,
    alpha=0.9
)
ax_map.text(
    zhangliang.lon.iloc[0] + 0.06, zhangliang.lat.iloc[0], 
    '张良镇',
    fontsize=CONFIG['zhangliang_font'], 
    fontweight='bold', 
    color=CONFIG['color_zhangliang'], 
    zorder=7,
    bbox=dict(boxstyle="round,pad=0.08", facecolor='white', alpha=0.9, edgecolor='none')
)

# 5. 地图样式优化
ax_map.set_xlim(henan_cities.total_bounds[0]-0.25, henan_cities.total_bounds[2]+0.25)
ax_map.set_ylim(henan_cities.total_bounds[1]-0.25, henan_cities.total_bounds[3]+0.25)
ax_map.set_axis_off()
ax_map.text(
    0.5, 1.01, "河南省·城市与特色美食分布", 
    transform=ax_map.transAxes,
    fontsize=16, 
    fontweight='bold', 
    ha='center', 
    color=CONFIG['color_border']
)

# ---------------------- 6. 绘制图例（原布局） ----------------------
ax_legend.set_axis_off()
ax_legend.text(
    0.5, 0.99, "特色美食图鉴", 
    fontsize=CONFIG['legend_title_font'],
    fontweight='bold', 
    ha='center', 
    color=CONFIG['color_border'],
    transform=ax_legend.transAxes
)

cols = 3
rows = 6
for i, (_, row) in enumerate(city_food_data.iterrows()):
    row_idx = i // cols
    col_idx = i % cols
    
    x = 0.03 + col_idx * CONFIG['legend_col_gap']
    y = 0.94 - row_idx * CONFIG['legend_row_gap']
    
    # 美食图片
    img = load_and_unify_img(row.food_img, CONFIG['img_zoom'])
    if img:
        ab = AnnotationBbox(
            img, (x + 0.12, y - 0.09),
            frameon=False, 
            xycoords='axes fraction',
            box_alignment=(0.5, 0.5)
        )
        ax_legend.add_artist(ab)
    
    # 市名
    ax_legend.text(
        x + 0.12, y, row.city, 
        fontsize=CONFIG['legend_text_font'],
        fontweight='bold', 
        ha='center', va='center', 
        color=CONFIG['color_city_text'],
        transform=ax_legend.transAxes
    )
    # 美食名
    ax_legend.text(
        x + 0.12, y - 0.14, row.food,
        fontsize=CONFIG['legend_text_font']-1,
        ha='center', va='center', 
        color=CONFIG['color_food_text'],
        transform=ax_legend.transAxes
    )

# ---------------------- 7. 保存文件 ----------------------
plt.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01)

output_png = "河南美食地图_精准调整版.png"
output_svg = "河南美食地图_精准调整版.svg"

# 保存PNG
plt.savefig(
    output_png,
    dpi=600,
    bbox_inches='tight',
    pad_inches=0.01,
    facecolor=CONFIG['color_bg'],
    edgecolor='none',
    pil_kwargs={'compression': 0, 'quality': 100}
)

# 保存SVG
plt.savefig(
    output_svg,
    format='svg',
    bbox_inches='tight',
    pad_inches=0.01,
    facecolor=CONFIG['color_bg'],
    edgecolor='none'
)

# ---------------------- 方案2：图片环绕地图版本（备选） ----------------------
print("\n✅ 正在生成图片环绕地图版本...")
fig2, ax2 = plt.subplots(figsize=(20, 16), dpi=600)
fig2.patch.set_facecolor(CONFIG['color_bg'])

# 绘制基础地图
henan_cities.plot(ax=ax2, facecolor='#f0f8ff', edgecolor='#cce5ff', linewidth=1.2, zorder=2, alpha=0.9)
henan_province.plot(ax=ax2, facecolor='none', edgecolor=CONFIG['color_border'], linewidth=3.5, zorder=4)

# 标注城市文字（同精准调整版）
for idx, row in henan_cities.iterrows():
    centroid = row.geometry.centroid
    city_name = row['city_name'] if 'city_name' in row else row['name']
    lon = centroid.x + CITY_ADJUST.get(city_name, {}).get('lon_offset', 0)
    lat = centroid.y + CITY_ADJUST.get(city_name, {}).get('lat_offset', 0)
    city_font_size = CONFIG['hebi_city_font'] if city_name == '鹤壁市' else CONFIG['city_font']
    food_font_size = CONFIG['hebi_food_font'] if city_name == '鹤壁市' else CONFIG['food_font']
    
    ax2.text(
        lon, lat + CONFIG['map_text_gap'], 
        city_name, 
        fontsize=city_font_size,
        ha='center', va='bottom', 
        fontweight='bold', 
        color=CONFIG['color_city_text'], 
        zorder=5,
        bbox=dict(boxstyle="round,pad=0.03", facecolor='white', alpha=0.7, edgecolor='none')
    )
    
    food_row = city_food_data[city_food_data['city'] == city_name]
    if not food_row.empty:
        food_name = food_row.iloc[0]['food']
        ax2.text(
            lon, lat - CONFIG['map_text_gap'], 
            f'({food_name})', 
            fontsize=food_font_size,
            ha='center', va='top', 
            color=CONFIG['color_food_text'], 
            zorder=5
        )

# 标注张良镇
ax2.scatter(zhangliang.lon, zhangliang.lat, s=CONFIG['zhangliang_ms'], c=CONFIG['color_zhangliang'], 
            edgecolor='white', linewidth=2, zorder=6, alpha=0.9)
ax2.text(zhangliang.lon.iloc[0] + 0.06, zhangliang.lat.iloc[0], '张良镇',
         fontsize=CONFIG['zhangliang_font'], fontweight='bold', color=CONFIG['color_zhangliang'], 
         zorder=7, bbox=dict(boxstyle="round,pad=0.08", facecolor='white', alpha=0.9, edgecolor='none'))

# 计算环绕坐标（圆形分布）
map_center_x = henan_province.total_bounds[[0,2]].mean()
map_center_y = henan_province.total_bounds[[1,3]].mean()
radius = 1.8  # 环绕半径
num_images = len(city_food_data)
angles = np.linspace(0, 2*np.pi, num_images, endpoint=False)

# 绘制环绕的美食图片
for i, (_, row) in enumerate(city_food_data.iterrows()):
    angle = angles[i]
    # 计算图片位置
    x = map_center_x + radius * np.cos(angle)
    y = map_center_y + radius * np.sin(angle)
    
    # 加载图片
    img = load_and_unify_img(row.food_img, 0.18)
    if img:
        ab = AnnotationBbox(
            img, (x, y),
            frameon=True,
            bboxprops=dict(boxstyle="round,pad=0.1", facecolor='white', edgecolor=CONFIG['color_border'], alpha=0.9),
            xycoords='data',
            box_alignment=(0.5, 0.5)
        )
        ax2.add_artist(ab)
    
    # 标注图片下方的文字
    ax2.text(x, y - 0.15, f"{row.city}\n{row.food}", 
             fontsize=9, ha='center', va='top', fontweight='bold',
             color=CONFIG['color_city_text'],
             bbox=dict(boxstyle="round,pad=0.05", facecolor='white', alpha=0.8, edgecolor='none'))

# 地图样式
ax2.set_xlim(map_center_x - 2.5, map_center_x + 2.5)
ax2.set_ylim(map_center_y - 2.0, map_center_y + 2.0)
ax2.set_axis_off()
ax2.text(0.5, 1.02, "河南省·城市与特色美食分布（图片环绕版）", 
         transform=ax2.transAxes, fontsize=18, fontweight='bold', ha='center', 
         color=CONFIG['color_border'])

# 保存环绕版本
plt.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
output_png2 = "河南美食地图_图片环绕版.png"
output_svg2 = "河南美食地图_图片环绕版.svg"

plt.savefig(output_png2, dpi=600, bbox_inches='tight', pad_inches=0.02,
            facecolor=CONFIG['color_bg'], edgecolor='none',
            pil_kwargs={'compression': 0, 'quality': 100})
plt.savefig(output_svg2, format='svg', bbox_inches='tight', pad_inches=0.02,
            facecolor=CONFIG['color_bg'], edgecolor='none')

plt.show()
print(f"\n✅ 所有版本已保存：")
print(f"   1. 精准调整版（原布局）：{os.path.abspath(output_png)}")
print(f"   2. 图片环绕版（备选）：{os.path.abspath(output_png2)}")
print(f"   （SVG矢量版已同步生成）")