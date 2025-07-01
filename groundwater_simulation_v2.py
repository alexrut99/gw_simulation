import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import exp1
import contextily as ctx
from matplotlib_scalebar.scalebar import ScaleBar
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from scipy.interpolate import RegularGridInterpolator
import math
import io
import streamlit as st
from datetime import time as datetime_time
import time
import os
import requests
from io import BytesIO
import matplotlib as mpl
import matplotlib.font_manager as fm
import base64
import logging
from datetime import datetime
import tempfile
from matplotlib.font_manager import FontProperties

# ตั้งค่าฟอนต์ภาษาไทย
def install_thai_font():
    """ติดตั้งฟอนต์ภาษาไทยสำหรับ Matplotlib"""
    try:
        # ตรวจสอบว่าฟอนต์ถูกติดตั้งแล้วหรือยัง
        font_names = [f.name for f in fm.fontManager.ttflist]
        
        # รายการฟอนต์ภาษาไทยที่รองรับ
        thai_fonts = [
            'TH Sarabun New',
            'TH Niramit AS',
            'TH Krub',
            'TH K2D July8',
            'TH KoHo',
            'TH Srisakdi',
            'Tahoma',
            'Garuda',
            'Umpush',
            'Loma',
            'Norasi',
            'Sawasdee',
            'Waree',
            'Angsana New',
            'Cordia New',
            'Browallia New'
        ]
        
        # ตรวจสอบว่ามีฟอนต์ภาษาไทยใดติดตั้งแล้วบ้าง
        installed_thai_fonts = [f for f in thai_fonts if f in font_names]
        
        if installed_thai_fonts:
            # ใช้ฟอนต์ภาษาไทยตัวแรกที่พบ
            selected_font = installed_thai_fonts[0]
            mpl.rcParams['font.family'] = selected_font
            mpl.rcParams['font.size'] = 10
            st.session_state['thai_font'] = selected_font
            st.info(f"ใช้ฟอนต์ภาษาไทย: {selected_font}")
            return selected_font
        
        st.warning("ไม่พบฟอนต์ภาษาไทยที่ติดตั้งไว้ กำลังดาวน์โหลดฟอนต์ TH Sarabun New...")
        
        # URL ของฟอนต์ TH Sarabun New
        font_url = "https://github.com/google/fonts/raw/main/ofl/sarabun/THSarabunNew.ttf"
        
        # ดาวน์โหลดฟอนต์
        response = requests.get(font_url)
        if response.status_code != 200:
            st.error("ไม่สามารถดาวน์โหลดฟอนต์ได้")
            raise Exception(f"HTTP error: {response.status_code}")
            
        # สร้างไดเรกทอรีชั่วคราว
        os.makedirs("temp_fonts", exist_ok=True)
        font_path = os.path.join("temp_fonts", "THSarabunNew.ttf")
        
        with open(font_path, "wb") as f:
            f.write(response.content)
        
        # เพิ่มฟอนต์ให้กับ matplotlib
        fm.fontManager.addfont(font_path)
        
        # ตั้งค่าฟอนต์เริ่มต้น
        mpl.rcParams['font.family'] = 'TH Sarabun New'
        mpl.rcParams['font.size'] = 10
        st.session_state['thai_font'] = 'TH Sarabun New'
        st.session_state['thai_font_path'] = font_path
        
        # อัปเดตแคชฟอนต์
        mpl.font_manager._rebuild()
        
        st.success("ติดตั้งฟอนต์ TH Sarabun New เรียบร้อยแล้ว")
        return 'TH Sarabun New'
        
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการติดตั้งฟอนต์: {str(e)}")
        # ใช้ฟอนต์สำรอง
        mpl.rcParams['font.family'] = 'Tahoma'
        mpl.rcParams['font.size'] = 10
        st.session_state['thai_font'] = 'Tahoma'
        return 'Tahoma'

# ตั้งค่าพารามิเตอร์พื้นฐาน
BUFFER = 200
DAYS_TO_SIMULATE = 7
MAX_HOURS = 24 * DAYS_TO_SIMULATE

# UTM Zone dictionary
utm_zone_dict = {
    '47N': 32647,
    '48N': 32648,
    '49N': 32649,
    '50N': 32650,
    '51N': 32651
}

def read_well_data(file_content, file_type):
    """อ่านข้อมูลบ่อน้ำจาก Excel หรือ CSV"""
    try:
        if file_type == 'excel':
            df = pd.read_excel(io.BytesIO(file_content))
        else:
            encodings = ['utf-8', 'tis-620', 'cp874', 'latin1']
            for encoding in encodings:
                try:
                    df = pd.read_csv(io.BytesIO(file_content), encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                st.error("ไม่สามารถอ่านไฟล์ CSV ได้ โปรดตรวจสอบ encoding")
                return []
        
        column_mapping = {
            'well_id': 'Well_ID', 'wellid': 'Well_ID', 'id': 'Well_ID',
            'easting': 'Easting', 'east': 'Easting', 'x': 'Easting',
            'northing': 'Northing', 'north': 'Northing', 'y': 'Northing',
            'q': 'Q', 'discharge': 'Q', 'pumping_rate': 'Q',
            't': 'T', 'transmissivity': 'T',
            's': 'S', 'storativity': 'S', 'storage': 'S',
            'status': 'Status', 'type': 'Status'
        }
        
        df.columns = df.columns.str.strip().str.lower()
        df = df.rename(columns=lambda x: column_mapping.get(x, x))
        
        required_columns = ['Well_ID', 'Easting', 'Northing', 'Q', 'T', 'S', 'Status']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error("ไฟล์ข้อมูลขาดคอลัมน์สำคัญ! คอลัมน์ที่ขาด:")
            st.error(missing_columns)
            st.error("คอลัมน์ที่มีในไฟล์:")
            st.error(df.columns.tolist())
            return []
        
        df['Well_ID'] = df['Well_ID'].astype(str).str.strip()
        
        numeric_cols = ['Easting', 'Northing', 'Q', 'T', 'S']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                mean_val = df[col].mean()
                df[col] = df[col].fillna(mean_val)
                st.warning(f"เติมค่าขาดหายในคอลัมน์ {col} ด้วยค่าเฉลี่ย: {mean_val:.4f}")
        
        status_mapping = {
            'pump': 'Pumping', 'pumping': 'Pumping',
            'obs': 'Observation', 'observation': 'Observation',
            'ob': 'Observation', 'monitor': 'Observation'
        }
        
        df['Status'] = df['Status'].astype(str).str.strip().str.lower()
        df['Status'] = df['Status'].map(status_mapping).fillna('Observation')
        
        invalid_wells = []
        for idx, row in df.iterrows():
            issues = []
            if row['T'] <= 0:
                issues.append(f"T={row['T']} (ต้อง > 0)")
            if row['S'] <= 0:
                issues.append(f"S={row['S']} (ต้อง > 0)")
            if row['Q'] < 0:
                issues.append(f"Q={row['Q']} (ต้อง >= 0)")
            
            if issues:
                invalid_wells.append(f"บ่อ {row['Well_ID']}: {', '.join(issues)}")
        
        if invalid_wells:
            st.warning("พบปัญหากับข้อมูลบ่อน้ำต่อไปนี้:")
            for issue in invalid_wells:
                st.write(f"  - {issue}")
            st.warning("โปรดตรวจสอบและแก้ไขข้อมูล")
        
        wells = [
            (row['Well_ID'], row['Easting'], row['Northing'], 
             max(row['Q'], 0), max(row['T'], 1e-10), max(row['S'], 1e-10), row['Status'])
            for _, row in df.iterrows()
        ]
        
        st.success(f"นำเข้าข้อมูลสำเร็จ: พบบ่อน้ำทั้งหมด {len(wells)} บ่อ")
        st.info(f"  - บ่อสูบน้ำ: {sum(1 for w in wells if w[6] == 'Pumping')} บ่อ")
        st.info(f"  - บ่อสังเกตการณ์: {sum(1 for w in wells if w[6] == 'Observation')} บ่อ")
        
        return wells
        
    except Exception as e:
        st.error(f"เกิดข้อผิดพลาดในการอ่านไฟล์: {str(e)}")
        return []

def create_grid(wells, num_points=100):
    if len(wells) == 0:
        x = np.linspace(0, 1000, num_points)
        y = np.linspace(0, 1000, num_points)
        return np.meshgrid(x, y)
    
    eastings = [w[1] for w in wells]
    northings = [w[2] for w in wells]
    
    min_easting, max_easting = min(eastings), max(eastings)
    min_northing, max_northing = min(northings), max(northings)
    
    x = np.linspace(min_easting - BUFFER, max_easting + BUFFER, num_points)
    y = np.linspace(min_northing - BUFFER, max_northing + BUFFER, num_points)
    return np.meshgrid(x, y)

def calculate_drawdown_at_time(X, Y, wells, t_hours, well_pumping_schedules, utm_epsg):
    if t_hours <= 0 or len(wells) == 0:
        return np.zeros_like(X)
    
    total_drawdown = np.zeros_like(X)
    n_days = int(np.ceil(t_hours / 24))
    
    for well in wells:
        well_id, x_well, y_well, Q, T, S, status = well
        if status != 'Pumping' or Q == 0 or T <= 0 or S <= 0:
            continue
            
        pumping_schedule = well_pumping_schedules.get(well_id, [])
        if not pumping_schedule:
            continue
            
        r = np.sqrt((X - x_well)**2 + (Y - y_well)**2)
        r = np.maximum(r, 1e-10)
        
        well_drawdown = np.zeros_like(X)
        
        for day in range(n_days):
            for start_hour, end_hour in pumping_schedule:
                t_start = day * 24 + start_hour
                t_end = day * 24 + end_hour
                
                tau_start = np.maximum(t_hours - t_start, 0)
                tau_end = np.maximum(t_hours - t_end, 0)
                
                u_start = np.zeros_like(r)
                u_end = np.zeros_like(r)
                
                mask_start = tau_start > 0
                if mask_start.any():
                    u_start[mask_start] = (r[mask_start]**2 * S) / (4 * T * tau_start[mask_start])
                    u_start = np.maximum(u_start, 1e-20)
                
                mask_end = tau_end > 0
                if mask_end.any():
                    u_end[mask_end] = (r[mask_end]**2 * S) / (4 * T * tau_end[mask_end])
                    u_end = np.maximum(u_end, 1e-20)
                
                W_u_start = np.zeros_like(u_start)
                W_u_end = np.zeros_like(u_end)
                
                mask_u_start = u_start < 100
                if mask_u_start.any():
                    W_u_start[mask_u_start] = exp1(u_start[mask_u_start])
                
                mask_u_end = u_end < 100
                if mask_u_end.any():
                    W_u_end[mask_u_end] = exp1(u_end[mask_u_end])
                
                drawdown_start = np.where(tau_start > 0, (Q / (4 * np.pi * T)) * W_u_start, 0)
                drawdown_end = np.where(tau_end > 0, (Q / (4 * np.pi * T)) * W_u_end, 0)
                
                well_drawdown += (drawdown_start - drawdown_end)
        
        total_drawdown += well_drawdown
    
    return total_drawdown

def create_hourly_plot(hour, wells, grid_resolution, utm_zone, well_pumping_schedules):
    """สร้างแผนภูมิระยะน้ำลดโดยไม่ใช้ข้อความภาษาไทยในแผนที่"""
    # ตั้งค่าฟอนต์พื้นฐาน
    plt.rcParams['font.family'] = 'sans-serif'
    
    # กำหนด EPSG จาก UTM zone
    utm_epsg = utm_zone_dict[utm_zone]
    
    # ปรับความละเอียดกริด
    if grid_resolution == 'สูง (200x200)':
        X, Y = create_grid(wells, 200)
    elif grid_resolution == 'กลาง (100x100)':
        X, Y = create_grid(wells, 100)
    else:
        X, Y = create_grid(wells, 50)
    
    # คำนวณระยะน้ำลด
    drawdown = calculate_drawdown_at_time(X, Y, wells, hour, well_pumping_schedules, utm_epsg)
    
    # สร้าง interpolator
    x_grid = X[0, :]
    y_grid = Y[:, 0]
    interp = RegularGridInterpolator((y_grid, x_grid), drawdown, method='linear')
    
    # หาค่าระดับสี
    max_drawdown = np.nanmax(drawdown) if len(wells) > 0 else 0
    min_drawdown = np.nanmin(drawdown) if len(wells) > 0 else 0
    
    if max_drawdown > 0:
        max_level = int(np.ceil(max_drawdown))
        
        if max_level < 1:
            levels = np.linspace(0, 1, 11)
            fmt = ticker.StrMethodFormatter("{x:.1f}")
            contour_fmt = '%1.1f'
        else:
            levels = np.arange(0, max_level + 1, 1)
            fmt = ticker.FuncFormatter(lambda x, pos: f'{int(x)}')
            contour_fmt = '%1.0f'
    else:
        levels = 10
        fmt = None
        contour_fmt = '%1.0f'
    
    # สร้าง figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # พล็อต contour
    if max_drawdown > 0:
        contour = ax.contourf(X, Y, drawdown, levels=levels, cmap='viridis_r', alpha=0.7)
        contour_lines = ax.contour(
            X, Y, drawdown, 
            levels=levels,
            colors='white',
            linewidths=0.8,
            alpha=0.7
        )
        
        ax.clabel(
            contour_lines, 
            inline=True,
            fontsize=10,
            fmt=contour_fmt,
            colors='white'
        )
        
        cbar = fig.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label('Drawdown (m)', fontsize=12, labelpad=10)
        cbar.ax.invert_yaxis()
        
        if fmt is not None:
            cbar.ax.yaxis.set_major_formatter(fmt)
            cbar.ax.tick_params(labelsize=10)
        
        max_idx = np.unravel_index(np.argmax(drawdown), drawdown.shape)
        max_x, max_y = X[max_idx], Y[max_idx]
        max_value = drawdown[max_idx]
        
        ax.plot(max_x, max_y, '*', color='gold', markersize=18, 
                markeredgecolor='red', markeredgewidth=1.5, zorder=20)
        
        ax.text(max_x, max_y + 50, 
                f'Max: {max_value:.2f} m', 
                fontsize=12, color='white', weight='bold',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.8, edgecolor='white'),
                zorder=21)
    else:
        max_value = 0.0
    
    # พล็อตบ่อน้ำ
    for well in wells:
        well_id, x_well, y_well, Q, T, S, status_val = well
            
        color = 'red' if status_val == 'Pumping' else 'blue'
        marker = 'o' if status_val == 'Pumping' else 's'
        
        pumping_now = False
        if status_val == 'Pumping' and well_id in well_pumping_schedules:
            hour_in_day = hour % 24
            for start_hour, end_hour in well_pumping_schedules[well_id]:
                if start_hour <= hour_in_day < end_hour:
                    pumping_now = True
                    break
        
        if pumping_now:
            ax.plot(x_well, y_well, marker, color='yellow', markersize=12, 
                    markeredgecolor='red', markeredgewidth=1.5, zorder=10)
        else:
            ax.plot(x_well, y_well, marker, color=color, markersize=10, 
                    markeredgecolor='white', markeredgewidth=1.5, zorder=10)
        
        # แสดง ID บ่อเป็นตัวเลข
        ax.text(x_well + 20, y_well + 20, str(well_id),
                fontsize=10, color='white', weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='orange' if pumping_now else color, alpha=0.9),
                zorder=11)
        
        # แสดงค่าระยะน้ำลด
        point = np.array([[y_well, x_well]])
        well_dd = interp(point)[0] if max_drawdown > 0 else 0.0
        ax.text(x_well, y_well - 30, f'{well_dd:.2f} m',
                fontsize=9, color='white', weight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='purple' if status_val == 'Pumping' else 'green', alpha=0.9),
                zorder=12)
    
    # กำหนดขอบเขต
    if len(wells) > 0:
        eastings = [w[1] for w in wells]
        northings = [w[2] for w in wells]
        min_easting, max_easting = min(eastings), max(eastings)
        min_northing, max_northing = min(northings), max(northings)
        ax.set_xlim(min_easting - BUFFER, max_easting + BUFFER)
        ax.set_ylim(min_northing - BUFFER, max_northing + BUFFER)
    else:
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)
    
    # ตั้งค่าแกนพิกัด
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
    
    # เพิ่มแผนที่ฐาน
    try:
        ctx.add_basemap(
            ax,
            crs=f'EPSG:{utm_epsg}',
            source=ctx.providers.Esri.WorldImagery,
            alpha=0.8
        )
    except Exception as e:
        st.warning(f"ไม่สามารถโหลดแผนที่ฐานได้: {str(e)}")
    
    # เพิ่ม scale bar
    scalebar = ScaleBar(
        1, 'm', 
        length_fraction=0.25,
        location='lower left',
        pad=0.5,
        color='white',
        box_color='black',
        box_alpha=0.5,
        font_properties={'size': 10}
    )
    ax.add_artist(scalebar)
    
    # ตั้งค่าชื่อแผนที่ - ใช้ภาษาอังกฤษเพื่อหลีกเลี่ยงปัญหา
    day_number = (hour // 24) + 1
    title = f'Drawdown at Hour {hour} (Day {day_number})\n'
    title += f'Max Drawdown: {max_value:.2f} m | Min Drawdown: {min_drawdown:.2f} m\n'
    title += f'UTM Zone: {utm_zone} (EPSG: {utm_epsg})'
    
    ax.set_title(title, fontsize=14, pad=20)
    ax.set_xlabel('Easting [m]', fontsize=12, labelpad=10)
    ax.set_ylabel('Northing [m]', fontsize=12, labelpad=10)
    ax.grid(alpha=0.3, linestyle='--', color='white')
    ax.set_aspect('equal')
    
    # เพิ่มคำอธิบายสัญลักษณ์ (ภาษาอังกฤษ)
    legend_patches = [
        mpatches.Patch(color='red', label='Pumping Well (Off)'),
        mpatches.Patch(color='yellow', label='Pumping Well (On)'),
        mpatches.Patch(color='blue', label='Observation Well'),
        plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
                  markersize=15, markeredgecolor='red', label='Max Drawdown Point'),
        mpatches.Patch(color='purple', label='Drawdown at Pumping Well'),
        mpatches.Patch(color='green', label='Drawdown at Observation Well')
    ]
    
    ax.legend(
        handles=legend_patches, 
        loc='upper right', 
        fontsize=10,
        framealpha=0.9
    )
    
    return fig, max_value, min_drawdown

def time_to_hours(t):
    """แปลงเวลาเป็นชั่วโมง (float)"""
    return t.hour + t.minute / 60

def get_pumping_schedules(pumping_controls):
    schedules = {}
    for well_id, controls in pumping_controls.items():
        periods = []
        
        if controls['morning_active'] and controls['morning_start'] <= controls['morning_end']:
            periods.append((
                time_to_hours(controls['morning_start']), 
                time_to_hours(controls['morning_end'])
            ))
        
        if controls['afternoon_active'] and controls['afternoon_start'] <= controls['afternoon_end']:
            periods.append((
                time_to_hours(controls['afternoon_start']), 
                time_to_hours(controls['afternoon_end'])
            ))
        
        if controls['evening_active'] and controls['evening_start'] <= controls['evening_end']:
            periods.append((
                time_to_hours(controls['evening_start']), 
                time_to_hours(controls['evening_end'])
            ))
        
        schedules[well_id] = periods
    
    return schedules

def main():
    st.set_page_config(
        page_title="การจำลองระยะน้ำลดในชั้นน้ำบาดาล",
        page_icon="💧",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("💧 การจำลองระยะน้ำลดในชั้นน้ำบาดาล")
    st.markdown("""
    แอปพลิเคชันนี้ใช้สำหรับจำลองระยะน้ำลดในชั้นน้ำบาดาลโดยอาศัยหลักการของ Theis equation 
    และการซ้อนทับ (superposition) สำหรับบ่อสูบน้ำหลายบ่อ
    """)
    
    # ติดตั้งฟอนต์ภาษาไทย
    if 'thai_font' not in st.session_state:
        thai_font = install_thai_font()
        st.session_state['thai_font'] = thai_font
    
    # แสดงข้อมูลฟอนต์ที่ใช้งาน
    thai_font = st.session_state.get('thai_font', 'Tahoma')
    st.sidebar.markdown(f"**ฟอนต์ที่ใช้งาน:** {thai_font}")
    
    # ทดสอบแสดงข้อความภาษาไทยใน Streamlit
    if st.sidebar.checkbox("ทดสอบแสดงภาษาไทยใน Streamlit"):
        st.subheader("ทดสอบการแสดงผลภาษาไทย")
        st.write("นี่คือข้อความภาษาไทยทดสอบ: สวัสดีประเทศไทย")
        st.write("กระทรวงทรัพยากรธรรมชาติและสิ่งแวดล้อม")
        st.write("กรมทรัพยากรน้ำบาดาล")
    
    # กำหนด session state
    if 'wells' not in st.session_state:
        st.session_state.wells = []
    
    if 'pumping_controls' not in st.session_state:
        st.session_state.pumping_controls = {}
    
    # ส่วนอัปโหลดไฟล์
    with st.expander("📤 อัปโหลดข้อมูลบ่อน้ำ", expanded=True):
        uploaded_file = st.file_uploader(
            "เลือกไฟล์ข้อมูลบ่อน้ำ (Excel หรือ CSV)",
            type=["xlsx", "xls", "csv"],
            accept_multiple_files=False
        )
        
        if uploaded_file:
            file_type = 'excel' if uploaded_file.name.endswith(('xlsx', 'xls')) else 'csv'
            st.session_state.wells = read_well_data(uploaded_file.getvalue(), file_type)
    
    # แสดงข้อมูลบ่อน้ำ (ถ้ามี)
    if st.session_state.wells:
        st.subheader("ข้อมูลบ่อน้ำ")
        well_df = pd.DataFrame(st.session_state.wells, 
                              columns=['Well_ID', 'Easting', 'Northing', 'Q', 'T', 'S', 'Status'])
        st.dataframe(well_df)
    
    # ส่วนควบคุมหลัก
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # ส่วนเลือก UTM Zone
        utm_zone = st.selectbox(
            "UTM Zone:",
            options=list(utm_zone_dict.keys()),
            index=1
        )
        
        # ส่วนเลือกความละเอียดกริด
        grid_resolution = st.selectbox(
            "ความละเอียดกริด:",
            options=['ต่ำ (50x50)', 'กลาง (100x100)', 'สูง (200x200)'],
            index=1
        )
        
        # ส่วนเลือกชั่วโมงการจำลอง
        hour = st.slider(
            "ชั่วโมงการจำลอง:",
            min_value=0,
            max_value=MAX_HOURS,
            value=0,
            step=1
        )
        
        # คำนวณวันที่และชั่วโมง
        day_number = (hour // 24) + 1
        hour_in_day = hour % 24
        
        # แสดงสถานะการสูบน้ำ (ภาษาไทย)
        status_html = f"<div style='background-color:#e8f4f8; padding:10px; border-radius:5px; border-left:5px solid #1f618d;'>"
        status_html += f"<b>วันที่ {day_number} | ชั่วโมง: {hour_in_day:.1f}</b><br>"
        
        pumping_wells = []
        inactive_wells = []
        
        for well in st.session_state.wells:
            well_id = well[0]
            if well[6] != 'Pumping':
                continue
                
            if well_id in st.session_state.pumping_controls:
                controls = st.session_state.pumping_controls[well_id]
                
                if not controls['active']:
                    inactive_wells.append(well_id)
                    continue
                
                is_pumping = False
                
                if controls['morning_active']:
                    start = time_to_hours(controls['morning_start'])
                    end = time_to_hours(controls['morning_end'])
                    if start <= hour_in_day < end:
                        is_pumping = True
                
                if controls['afternoon_active']:
                    start = time_to_hours(controls['afternoon_start'])
                    end = time_to_hours(controls['afternoon_end'])
                    if start <= hour_in_day < end:
                        is_pumping = True
                
                if controls['evening_active']:
                    start = time_to_hours(controls['evening_start'])
                    end = time_to_hours(controls['evening_end'])
                    if start <= hour_in_day < end:
                        is_pumping = True
                
                if is_pumping:
                    pumping_wells.append(well_id)
        
        if pumping_wells:
            status_html += f"<span style='color:#27ae60; font-weight:bold;'>● กำลังสูบน้ำในบ่อ: {', '.join(pumping_wells)}</span><br>"
        else:
            status_html += f"<span style='color:#7f8c8d;'>● ไม่มีบ่อสูบน้ำที่ทำงาน</span><br>"
        
        if inactive_wells:
            status_html += f"<span style='color:#e67e22;'>● บ่อที่ปิดการใช้งาน: {', '.join(inactive_wells)}</span>"
        
        status_html += "</div>"
        
        st.markdown(status_html, unsafe_allow_html=True)
        
        # ปุ่มอัปเดต
        if st.button("อัปเดตการจำลอง", type="primary", use_container_width=True):
            st.rerun()
    
    with col2:
        # ส่วนตั้งค่าการสูบน้ำสำหรับแต่ละบ่อ
        if st.session_state.wells:
            st.subheader("ตั้งค่าการสูบน้ำสำหรับแต่ละบ่อ")
            
            # กรองเฉพาะบ่อสูบน้ำ
            pumping_wells = [well for well in st.session_state.wells if well[6] == 'Pumping']
            
            if not pumping_wells:
                st.warning("ไม่พบบ่อสูบน้ำในข้อมูล")
            else:
                # สร้างแท็บสำหรับแต่ละบ่อ
                tabs = st.tabs([f"บ่อ {well[0]}" for well in pumping_wells])
                
                for i, well in enumerate(pumping_wells):
                    well_id, x, y, Q, T, S, _ = well
                    
                    with tabs[i]:
                        # ตรวจสอบว่ามีข้อมูลควบคุมหรือไม่
                        if well_id not in st.session_state.pumping_controls:
                            st.session_state.pumping_controls[well_id] = {
                                'active': True,
                                'morning_active': True,
                                'morning_start': datetime_time(8, 0),
                                'morning_end': datetime_time(12, 0),
                                'afternoon_active': True,
                                'afternoon_start': datetime_time(12, 0),
                                'afternoon_end': datetime_time(18, 0),
                                'evening_active': True,
                                'evening_start': datetime_time(18, 0),
                                'evening_end': datetime_time(23, 59),
                            }
                        
                        controls = st.session_state.pumping_controls[well_id]
                        
                        # แสดงข้อมูลบ่อ (ภาษาไทย)
                        st.markdown(f"""
                        **ข้อมูลบ่อ {well_id}**
                        - ตำแหน่ง: ({x:.2f}, {y:.2f})
                        - อัตราการสูบ (Q): {Q:.2f} m³/day
                        - ค่าการนำน้ำ (T): {T:.4f} m²/day
                        - ค่าความสามารถในการกักเก็บ (S): {S:.6f}
                        """)
                        
                        controls['active'] = st.checkbox(
                            "เปิดใช้งานบ่อนี้", 
                            value=controls['active'],
                            key=f"active_{well_id}"
                        )
                        
                        st.markdown("**ช่วงเช้า (00:00-12:00 น.)**")
                        col_m1, col_m2 = st.columns(2)
                        with col_m1:
                            controls['morning_active'] = st.checkbox(
                                "เปิดใช้งานช่วงเช้า", 
                                value=controls['morning_active'],
                                key=f"morning_active_{well_id}"
                            )
                        with col_m2:
                            st.write("เวลาเริ่ม - สิ้นสุด:")
                            controls['morning_start'] = st.time_input(
                                "เวลาเริ่ม", 
                                value=controls['morning_start'],
                                key=f"morning_start_{well_id}",
                                label_visibility="collapsed"
                            )
                            controls['morning_end'] = st.time_input(
                                "เวลาสิ้นสุด", 
                                value=controls['morning_end'],
                                key=f"morning_end_{well_id}",
                                label_visibility="collapsed"
                            )
                        
                        st.markdown("**ช่วงบ่าย (12:00-18:00 น.)**")
                        col_a1, col_a2 = st.columns(2)
                        with col_a1:
                            controls['afternoon_active'] = st.checkbox(
                                "เปิดใช้งานช่วงบ่าย", 
                                value=controls['afternoon_active'],
                                key=f"afternoon_active_{well_id}"
                            )
                        with col_a2:
                            st.write("เวลาเริ่ม - สิ้นสุด:")
                            controls['afternoon_start'] = st.time_input(
                                "เวลาเริ่ม", 
                                value=controls['afternoon_start'],
                                key=f"afternoon_start_{well_id}",
                                label_visibility="collapsed"
                            )
                            controls['afternoon_end'] = st.time_input(
                                "เวลาสิ้นสุด", 
                                value=controls['afternoon_end'],
                                key=f"afternoon_end_{well_id}",
                                label_visibility="collapsed"
                            )
                        
                        st.markdown("**ช่วงเย็น (18:00-24:00 น.)**")
                        col_e1, col_e2 = st.columns(2)
                        with col_e1:
                            controls['evening_active'] = st.checkbox(
                                "เปิดใช้งานช่วงเย็น", 
                                value=controls['evening_active'],
                                key=f"evening_active_{well_id}"
                            )
                        with col_e2:
                            st.write("เวลาเริ่ม - สิ้นสุด:")
                            controls['evening_start'] = st.time_input(
                                "เวลาเริ่ม", 
                                value=controls['evening_start'],
                                key=f"evening_start_{well_id}",
                                label_visibility="collapsed"
                            )
                            controls['evening_end'] = st.time_input(
                                "เวลาสิ้นสุด", 
                                value=controls['evening_end'],
                                key=f"evening_end_{well_id}",
                                label_visibility="collapsed"
                            )
    
    # สร้างแผนที่และแสดงคำอธิบายภาษาไทยใน Streamlit
    if st.session_state.wells:
        st.subheader("ผลลัพธ์การจำลอง")
        
        # ดึงตารางเวลาการสูบน้ำ
        pumping_schedules = get_pumping_schedules(st.session_state.pumping_controls)
        
        # สร้างแผนที่
        with st.spinner('กำลังคำนวณและสร้างแผนที่...'):
            start_time = time.time()
            fig, max_value, min_drawdown = create_hourly_plot(
                hour,
                st.session_state.wells,
                grid_resolution,
                utm_zone,
                pumping_schedules
            )
            st.pyplot(fig)
            elapsed = time.time() - start_time
            
            # แสดงผลภาษาไทยใน Streamlit
            day_number = (hour // 24) + 1
            st.success(f"คำนวณเสร็จสิ้นใน {elapsed:.2f} วินาที")
            st.subheader(f"สรุปผลการจำลอง ณ ชั่วโมง {hour} (วันที่ {day_number})")
            st.markdown(f"""
            **ผลการจำลอง:**
            - ระยะน้ำลดสูงสุด: **{max_value:.2f} เมตร**
            - ระยะน้ำลดต่ำสุด: **{min_drawdown:.2f} เมตร**
            - โซน UTM: **{utm_zone}** (EPSG: {utm_zone_dict[utm_zone]})
            """)
            
            # คำอธิบายสัญลักษณ์ภาษาไทย
            st.subheader("คำอธิบายสัญลักษณ์ในแผนที่")
            st.markdown("""
            <style>
            .legend-box {
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 15px;
                margin-bottom: 20px;
                background-color: #f9f9f9;
            }
            .legend-item {
                display: flex;
                align-items: center;
                margin-bottom: 8px;
            }
            .color-box {
                width: 20px;
                height: 20px;
                margin-right: 10px;
                border: 1px solid #555;
            }
            </style>
            
            <div class="legend-box">
                <div class="legend-item">
                    <div class="color-box" style="background-color: red;"></div>
                    <div>บ่อสูบน้ำ (หยุดทำงาน)</div>
                </div>
                <div class="legend-item">
                    <div class="color-box" style="background-color: yellow;"></div>
                    <div>บ่อสูบน้ำ (กำลังทำงาน)</div>
                </div>
                <div class="legend-item">
                    <div class="color-box" style="background-color: blue;"></div>
                    <div>บ่อสังเกตการณ์</div>
                </div>
                <div class="legend-item">
                    <div class="color-box" style="background-color: purple;"></div>
                    <div>ค่าระยะน้ำลดที่บ่อสูบน้ำ</div>
                </div>
                <div class="legend-item">
                    <div class="color-box" style="background-color: green;"></div>
                    <div>ค่าระยะน้ำลดที่บ่อสังเกตการณ์</div>
                </div>
                <div class="legend-item">
                    <div style="font-size: 24px; color: gold; text-shadow: 0 0 2px red;">★</div>
                    <div style="margin-left: 10px;">ตำแหน่งที่มีระยะน้ำลดสูงสุด</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # ปุ่มดาวน์โหลดภาพ
            buf = BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
            st.download_button(
                label="ดาวน์โหลดแผนที่",
                data=buf,
                file_name=f"groundwater_simulation_hour_{hour}.png",
                mime="image/png",
                use_container_width=True
            )

if __name__ == "__main__":
    main()