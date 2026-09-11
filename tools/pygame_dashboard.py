import argparse
import glob
import os
import shutil
import time
import tkinter as tk
from tkinter import filedialog, ttk

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

pygame = None


PANEL_WIDTH = 320
WINDOW_SIZE = (1280, 720)
BACKGROUND = (18, 22, 28)
PANEL_BG = (28, 33, 41)
TEXT_MAIN = (235, 239, 245)
TEXT_MUTED = (160, 170, 182)
ACCENT = (114, 192, 255)
BUTTON_BG = (44, 52, 64)
BUTTON_BG_HOVER = (58, 68, 82)
BUTTON_BORDER = (82, 92, 108)
CHECK_ON = (91, 199, 128)
WARNING = (255, 194, 87)


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _find_agent_label(agent_id, base_dir):
    agent_dir = os.path.join(base_dir, 'agent_{}'.format(agent_id))
    if not os.path.isdir(agent_dir):
        return 'agent_{}'.format(agent_id)

    yaml_files = sorted(glob.glob(os.path.join(agent_dir, '*.yaml')))
    if not yaml_files:
        return 'agent_{}'.format(agent_id)

    try:
        with open(yaml_files[0], 'r') as handle:
            data = yaml.safe_load(handle) or {}
        policy = data.get('POLICY', {})
        algo = policy.get('NAME')
        if algo:
            return '{} (agent_{})'.format(algo, agent_id)
    except Exception:
        return 'agent_{}'.format(agent_id)

    return 'agent_{}'.format(agent_id)


def _scan_agents(base_dir, seed_ids=None):
    agent_ids = set(seed_ids or [])
    candidates = glob.glob(os.path.join(base_dir, 'agent_*'))
    for path in candidates:
        name = os.path.basename(path)
        if name.startswith('agent_'):
            try:
                agent_ids.add(int(name.split('_', 1)[1]))
            except ValueError:
                continue
    return sorted(agent_ids)


def _list_trajectory_files(agent_id, base_dir):
    traj_dir = os.path.join(base_dir, 'agent_{}'.format(agent_id), 'trajectories')
    if not os.path.isdir(traj_dir):
        return []
    candidates = glob.glob(os.path.join(traj_dir, '*.csv'))
    candidates += glob.glob(os.path.join(traj_dir, 'plans', '*.csv'))
    candidates += glob.glob(os.path.join(traj_dir, 'surrounding', '*.csv'))
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates


def _short_file_label(path):
    name = os.path.basename(path)
    parent = os.path.basename(os.path.dirname(path))
    if parent == 'plans':
        return 'plan: {}'.format(name)
    if parent == 'surrounding':
        return 'traffic: {}'.format(name)
    return 'ego: {}'.format(name)


def _agent_summary(agent_id, base_dir):
    agent_dir = os.path.join(base_dir, 'agent_{}'.format(agent_id))
    models_dir = os.path.join(agent_dir, 'models')
    traj_dir = os.path.join(agent_dir, 'trajectories')
    plans_dir = os.path.join(traj_dir, 'plans')
    monitor = _load_monitor(agent_id, base_dir)

    summary = {
        'agent_id': agent_id,
        'agent_dir': agent_dir,
        'has_monitor': monitor is not None,
        'episodes': 0,
        'timesteps': 0,
        'mean_tail_reward': 0.0,
        'max_reward': 0.0,
        'models': len(glob.glob(os.path.join(models_dir, '*.zip'))) if os.path.isdir(models_dir) else 0,
        'best_models': len(glob.glob(os.path.join(models_dir, 'best_*.zip'))) if os.path.isdir(models_dir) else 0,
        'step_models': len(glob.glob(os.path.join(models_dir, 'step_*.zip'))) if os.path.isdir(models_dir) else 0,
        'trajectory_files': len(glob.glob(os.path.join(traj_dir, 'trajectory_*.csv'))) if os.path.isdir(traj_dir) else 0,
        'plan_files': len(glob.glob(os.path.join(plans_dir, '*.csv'))) if os.path.isdir(plans_dir) else 0,
        'surrounding_files': len(glob.glob(os.path.join(traj_dir, 'surrounding', '*.csv'))) if os.path.isdir(traj_dir) else 0,
        'config_files': len(glob.glob(os.path.join(agent_dir, '*.yaml'))) if os.path.isdir(agent_dir) else 0,
    }

    if monitor is not None and len(monitor):
        rewards = monitor['r'].values
        lengths = monitor['l'].values
        tail = rewards[-100:] if len(rewards) >= 100 else rewards
        summary['episodes'] = int(len(monitor))
        summary['timesteps'] = int(np.sum(lengths))
        summary['mean_tail_reward'] = float(np.mean(tail)) if len(tail) else 0.0
        summary['max_reward'] = float(np.max(rewards)) if len(rewards) else 0.0

    return summary


def _trajectory_summary(csv_path):
    if not csv_path or not os.path.isfile(csv_path):
        return {}
    data = pd.read_csv(csv_path)
    summary = {'rows': int(len(data)), 'kind': 'unknown'}
    if 'path_type' in data.columns:
        candidates = data[data['path_type'] == 'candidate']
        selected = data[data['path_type'] == 'selected']
        summary.update({
            'kind': 'planner',
            'candidate_paths': int(candidates['traj_id'].nunique()) if not candidates.empty else 0,
            'selected_points': int(len(selected)),
        })
    elif 'role' in data.columns and 'actor_id' in data.columns:
        traffic = data[data['role'] == 'traffic']
        ego = data[data['role'] == 'ego']
        summary.update({
            'kind': 'surrounding',
            'ego_samples': int(len(ego)),
            'traffic_actors': int(traffic['actor_id'].nunique()) if not traffic.empty else 0,
            'traffic_samples': int(len(traffic)),
        })
    elif 'ego_x' in data.columns:
        summary.update({
            'kind': 'ego',
            'samples': int(len(data)),
            'lane_change_samples': int(data['lane_change'].sum()) if 'lane_change' in data.columns else 0,
            'mean_speed': float(data['ego_speed'].mean()) if 'ego_speed' in data.columns and len(data) else 0.0,
            'max_lead_dist': float(data['lead_dist'].max()) if 'lead_dist' in data.columns and data['lead_dist'].notna().any() else 0.0,
        })
    return summary


def _load_monitor(agent_id, base_dir):
    path = os.path.join(base_dir, 'agent_{}'.format(agent_id), 'monitor.csv')
    if not os.path.isfile(path):
        return None
    return pd.read_csv(path, skiprows=1)


def _moving_stats(values, window):
    if len(values) < window:
        return np.array([]), np.array([])
    means = []
    stds = []
    for i in range(window - 1, len(values)):
        window_slice = values[i - window + 1:i + 1]
        means.append(np.mean(window_slice))
        stds.append(np.std(window_slice))
    return np.array(means), np.array(stds)


def _plot_rewards(agent_ids, labels, window_size, base_dir, output_path):
    plt.figure(figsize=(10, 6))
    for idx, agent_id in enumerate(agent_ids):
        data = _load_monitor(agent_id, base_dir)
        if data is None:
            continue
        rewards = data['r'].values
        timesteps = np.cumsum(data['l'].values)
        means, stds = _moving_stats(rewards, window_size)
        if len(means) == 0:
            plt.plot(timesteps, rewards, alpha=0.6, label=labels[idx])
            continue
        x = timesteps[window_size - 1:]
        plt.plot(x, means, label=labels[idx])
        plt.fill_between(x, means - stds, means + stds, alpha=0.2)

    plt.title('Training Reward Curves')
    plt.xlabel('Timesteps')
    plt.ylabel('Reward (moving mean)')
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_trajectory(traj_path, output_path):
    data = pd.read_csv(traj_path)
    fig, ax = plt.subplots(figsize=(8.5, 8))
    if 'path_type' in data.columns:
        candidates = data[data['path_type'] == 'candidate']
        selected = data[data['path_type'] == 'selected']
        for traj_id in candidates['traj_id'].unique():
            traj = candidates[candidates['traj_id'] == traj_id]
            ax.plot(traj['x'], traj['y'], color='#9aa5b1', linewidth=1, alpha=0.32)
        if not selected.empty:
            ax.plot(selected['x'], selected['y'], color='#1464f4', linewidth=3.0, label='selected trajectory')
            ax.scatter(selected['x'].iloc[0], selected['y'].iloc[0], color='#31a354', s=50, zorder=5, label='start')
            ax.scatter(selected['x'].iloc[-1], selected['y'].iloc[-1], color='#ef3b2c', s=50, zorder=5, label='target')
        ax.set_title('Frenet Planner: Candidate Paths and Selected Trajectory')
        ax.text(0.02, 0.98,
                'candidate paths: {}\nselected points: {}'.format(
                    candidates['traj_id'].nunique() if not candidates.empty else 0,
                    len(selected)),
                transform=ax.transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.82, edgecolor='#cfd8dc'))
    else:
        if 'ego_speed' in data.columns and len(data) > 2:
            points = ax.scatter(data['ego_x'], data['ego_y'], c=data['ego_speed'],
                                cmap='viridis', s=18, label='ego samples')
            fig.colorbar(points, ax=ax, label='ego speed')
            ax.plot(data['ego_x'], data['ego_y'], color='#2c7fb8', linewidth=1.2, alpha=0.35)
        else:
            ax.plot(data['ego_x'], data['ego_y'], color='#2c7fb8', linewidth=2, label='ego')

        if 'lead_x' in data.columns and data['lead_x'].notna().any():
            ax.plot(data['lead_x'], data['lead_y'], color='#f03b20', linewidth=2, label='lead vehicle')

        if 'lane_change' in data.columns:
            change_points = data[data['lane_change'] > 0]
            if not change_points.empty:
                ax.scatter(change_points['ego_x'], change_points['ego_y'],
                           color='#31a354', s=28, label='lane change')

        ax.set_title('Ego Vehicle Tracking and Interaction Trace')
        if len(data):
            ax.text(0.02, 0.98,
                    'samples: {}\nmean speed: {:.2f} m/s'.format(
                        len(data),
                        data['ego_speed'].mean() if 'ego_speed' in data.columns else 0.0),
                    transform=ax.transAxes, va='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.82, edgecolor='#cfd8dc'))
    ax.set_xlabel('World X')
    ax.set_ylabel('World Y')
    ax.axis('equal')
    ax.grid(True, alpha=0.2)
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_surrounding_tracks(track_path, output_path, max_points_per_actor=500):
    data = pd.read_csv(track_path)
    if data.empty or 'role' not in data.columns:
        raise ValueError('Selected file is not a surrounding trajectory CSV.')

    fig, ax = plt.subplots(figsize=(9, 8))
    ego = data[data['role'] == 'ego']
    traffic = data[data['role'] == 'traffic']

    if not traffic.empty:
        cmap = plt.get_cmap('tab20')
        actor_ids = list(traffic['actor_id'].dropna().unique())
        for idx, actor_id in enumerate(actor_ids):
            actor_track = traffic[traffic['actor_id'] == actor_id].sort_values('step')
            if len(actor_track) > max_points_per_actor:
                actor_track = actor_track.iloc[-max_points_per_actor:]
            color = cmap(idx % 20)
            ax.plot(actor_track['x'], actor_track['y'], linewidth=1.4, alpha=0.72, color=color)
            ax.scatter(actor_track['x'].iloc[-1], actor_track['y'].iloc[-1], s=18, color=color)

    if not ego.empty:
        ego = ego.sort_values('step')
        if len(ego) > max_points_per_actor:
            ego = ego.iloc[-max_points_per_actor:]
        ax.plot(ego['x'], ego['y'], color='#0057ff', linewidth=3.2, label='ego vehicle')
        ax.scatter(ego['x'].iloc[0], ego['y'].iloc[0], color='#31a354', s=55, zorder=5, label='ego start')
        ax.scatter(ego['x'].iloc[-1], ego['y'].iloc[-1], color='#ef3b2c', s=55, zorder=5, label='ego current')

    ax.set_title('Ego and Surrounding Vehicle Trajectories')
    ax.set_xlabel('World X')
    ax.set_ylabel('World Y')
    ax.axis('equal')
    ax.grid(True, alpha=0.2)
    ax.text(0.02, 0.98,
            'ego samples: {}\ntraffic actors: {}\ntraffic samples: {}'.format(
                len(ego),
                traffic['actor_id'].nunique() if not traffic.empty else 0,
                len(traffic)),
            transform=ax.transAxes, va='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.84, edgecolor='#cfd8dc'))
    ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_agent_overview(agent_ids, labels, base_dir, output_path):
    summaries = [_agent_summary(agent_id, base_dir) for agent_id in agent_ids]
    if not summaries:
        return

    x = np.arange(len(summaries))
    model_counts = [s['models'] for s in summaries]
    traj_counts = [s['trajectory_files'] for s in summaries]
    plan_counts = [s['plan_files'] for s in summaries]
    surrounding_counts = [s['surrounding_files'] for s in summaries]
    rewards = [s['mean_tail_reward'] for s in summaries]
    timesteps = [s['timesteps'] for s in summaries]

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 0.95])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, :])

    width = 0.25
    ax0.bar(x - width, model_counts, width, color='#1565c0', label='models')
    ax0.bar(x, traj_counts, width, color='#2e7d32', label='ego trajectories')
    ax0.bar(x + width, plan_counts, width, color='#ef6c00', label='planner snapshots')
    ax0.plot(x, surrounding_counts, color='#7b1fa2', marker='o', linewidth=2, label='surrounding tracks')
    ax0.set_title('Training Artifacts')
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=18, ha='right')
    ax0.grid(True, axis='y', alpha=0.2)
    ax0.legend()

    ax1.scatter(timesteps, rewards, s=90, color='#7b1fa2')
    for i, label in enumerate(labels):
        ax1.annotate(label, (timesteps[i], rewards[i]), xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax1.set_title('Learning Progress Summary')
    ax1.set_xlabel('timesteps')
    ax1.set_ylabel('tail mean reward')
    ax1.grid(True, alpha=0.2)

    ax2.axis('off')
    rows = []
    for label, summary in zip(labels, summaries):
        rows.append([
            label,
            summary['episodes'],
            summary['timesteps'],
            '{:.2f}'.format(summary['mean_tail_reward']),
            '{:.2f}'.format(summary['max_reward']),
            summary['best_models'],
            summary['trajectory_files'],
            summary['plan_files'],
            summary['surrounding_files'],
        ])
    table = ax2.table(
        cellText=rows,
        colLabels=['agent', 'episodes', 'timesteps', 'tail reward', 'max reward',
                   'best models', 'ego csv', 'plan csv', 'traffic csv'],
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    ax2.set_title('Artifact Index', pad=12)

    fig.suptitle('RL-Frenet-CARLA Software Artifact Overview', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_comparison(agent_ids, labels, window_size, base_dir, output_path):
    means = []
    maxes = []
    for agent_id in agent_ids:
        data = _load_monitor(agent_id, base_dir)
        if data is None:
            means.append(0)
            maxes.append(0)
            continue
        rewards = data['r'].values
        tail = rewards[-window_size:] if len(rewards) >= window_size else rewards
        means.append(np.mean(tail) if len(tail) else 0)
        maxes.append(np.max(rewards) if len(rewards) else 0)

    x = np.arange(len(agent_ids))
    width = 0.38

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(x - width / 2, means, width, color='#4292c6', label='mean reward')
    axes[0].bar(x + width / 2, maxes, width, color='#fb6a4a', label='max reward')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=20, ha='right')
    axes[0].set_title('Reward Summary')
    axes[0].grid(True, axis='y', alpha=0.2)
    axes[0].legend()

    axes[1].plot(means, marker='o', color='#2c7fb8', label='mean reward')
    axes[1].plot(maxes, marker='s', color='#ef3b2c', label='max reward')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=20, ha='right')
    axes[1].set_title('Method Comparison')
    axes[1].grid(True, alpha=0.2)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _latest_trajectory(agent_id, base_dir):
    candidates = _list_trajectory_files(agent_id, base_dir)
    return candidates[0] if candidates else None


def _latest_surrounding(agent_id, base_dir):
    if agent_id is None:
        return None
    path = os.path.join(
        base_dir,
        'agent_{}'.format(agent_id),
        'trajectories',
        'surrounding',
        'latest_surrounding_tracks.csv'
    )
    if os.path.isfile(path):
        return path
    candidates = glob.glob(os.path.join(
        base_dir,
        'agent_{}'.format(agent_id),
        'trajectories',
        'surrounding',
        '*.csv'
    ))
    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0] if candidates else None


def _export_artifacts(agent_ids, base_dir, plots_dir, export_dir):
    _ensure_dir(export_dir)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    export_root = os.path.join(export_dir, 'export_{}'.format(stamp))
    _ensure_dir(export_root)

    for agent_id in agent_ids:
        agent_src = os.path.join(base_dir, 'agent_{}'.format(agent_id))
        if not os.path.isdir(agent_src):
            continue
        agent_dst = os.path.join(export_root, 'agent_{}'.format(agent_id))
        _ensure_dir(agent_dst)

        for name in ['monitor.csv', 'reproduction_info.txt']:
            src = os.path.join(agent_src, name)
            if os.path.isfile(src):
                shutil.copy2(src, agent_dst)

        for yaml_file in glob.glob(os.path.join(agent_src, '*.yaml')):
            shutil.copy2(yaml_file, agent_dst)

        models_src = os.path.join(agent_src, 'models')
        if os.path.isdir(models_src):
            models_dst = os.path.join(agent_dst, 'models')
            if os.path.isdir(models_dst):
                shutil.rmtree(models_dst)
            shutil.copytree(models_src, models_dst)

        traj_src = os.path.join(agent_src, 'trajectories')
        if os.path.isdir(traj_src):
            traj_dst = os.path.join(agent_dst, 'trajectories')
            if os.path.isdir(traj_dst):
                shutil.rmtree(traj_dst)
            shutil.copytree(traj_src, traj_dst)

    if os.path.isdir(plots_dir):
        plots_dst = os.path.join(export_root, 'plots')
        if os.path.isdir(plots_dst):
            shutil.rmtree(plots_dst)
        shutil.copytree(plots_dir, plots_dst)

    return export_root


def _ask_open_file(initial_dir):
    try:
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        file_path = filedialog.askopenfilename(initialdir=initial_dir,
                                               filetypes=[('CSV files', '*.csv')])
        root.destroy()
        return file_path or None
    except Exception:
        return None


def _load_image(path, target_rect):
    image = pygame.image.load(path).convert()
    image_rect = image.get_rect()
    scale = min(target_rect.width / image_rect.width, target_rect.height / image_rect.height)
    if scale < 1:
        new_size = (int(image_rect.width * scale), int(image_rect.height * scale))
        image = pygame.transform.smoothscale(image, new_size)
        image_rect = image.get_rect()
    image_rect.center = target_rect.center
    return image, image_rect


def _draw_text(surface, text, font, color, x, y):
    text_surface = font.render(text, True, color)
    surface.blit(text_surface, (x, y))


def _draw_wrapped_text(surface, text, font, color, rect, line_height=18):
    words = str(text).split()
    lines = []
    line = ''
    for word in words:
        probe = word if not line else '{} {}'.format(line, word)
        if font.size(probe)[0] <= rect.width:
            line = probe
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)

    y = rect.y
    for line in lines:
        if y + line_height > rect.bottom:
            break
        _draw_text(surface, line, font, color, rect.x, y)
        y += line_height


def _draw_info_card(surface, title, rows, rect, font, title_font):
    pygame.draw.rect(surface, BUTTON_BG, rect, border_radius=8)
    pygame.draw.rect(surface, BUTTON_BORDER, rect, width=1, border_radius=8)
    _draw_text(surface, title, title_font, TEXT_MAIN, rect.x + 12, rect.y + 10)
    y = rect.y + 40
    for key, value, color in rows:
        _draw_text(surface, key, font, TEXT_MUTED, rect.x + 12, y)
        value_surface = font.render(str(value), True, color)
        surface.blit(value_surface, (rect.right - 12 - value_surface.get_width(), y))
        y += 22


class Button:
    def __init__(self, rect, label, on_click):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.on_click = on_click
        self.hovered = False

    def draw(self, surface, font):
        bg = BUTTON_BG_HOVER if self.hovered else BUTTON_BG
        pygame.draw.rect(surface, bg, self.rect, border_radius=6)
        pygame.draw.rect(surface, BUTTON_BORDER, self.rect, width=1, border_radius=6)
        text_surface = font.render(self.label, True, TEXT_MAIN)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.on_click()


class Dropdown:
    def __init__(self, rect, options, selected_index=0):
        self.rect = pygame.Rect(rect)
        self.options = options
        self.selected_index = max(0, min(selected_index, len(options) - 1)) if options else 0
        self.expanded = False

    def set_options(self, options):
        self.options = options
        if not options:
            self.selected_index = 0
        else:
            self.selected_index = min(self.selected_index, len(options) - 1)

    def selected(self):
        if not self.options:
            return None
        return self.options[self.selected_index]

    def draw(self, surface, font):
        pygame.draw.rect(surface, BUTTON_BG, self.rect, border_radius=6)
        pygame.draw.rect(surface, BUTTON_BORDER, self.rect, width=1, border_radius=6)
        label = self.selected() or 'None'
        text_surface = font.render(label, True, TEXT_MAIN)
        surface.blit(text_surface, (self.rect.x + 8, self.rect.y + 6))
        pygame.draw.polygon(surface, TEXT_MUTED,
                            [(self.rect.right - 18, self.rect.y + 10),
                             (self.rect.right - 8, self.rect.y + 10),
                             (self.rect.right - 13, self.rect.y + 18)])

    def draw_overlay(self, surface, font):
        if not self.expanded or not self.options:
            return
        drop_rect = pygame.Rect(self.rect.x, self.rect.bottom + 4,
                                self.rect.width, min(200, 28 * len(self.options)))
        pygame.draw.rect(surface, PANEL_BG, drop_rect, border_radius=6)
        pygame.draw.rect(surface, BUTTON_BORDER, drop_rect, width=1, border_radius=6)
        for idx, option in enumerate(self.options):
            row_rect = pygame.Rect(drop_rect.x + 4, drop_rect.y + 4 + idx * 28,
                                   drop_rect.width - 8, 24)
            if row_rect.bottom > drop_rect.bottom:
                break
            if idx == self.selected_index:
                pygame.draw.rect(surface, BUTTON_BG_HOVER, row_rect, border_radius=4)
            text = font.render(option, True, TEXT_MAIN)
            surface.blit(text, (row_rect.x + 6, row_rect.y + 2))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.expanded = not self.expanded
                return True
            if self.expanded:
                drop_rect = pygame.Rect(self.rect.x, self.rect.bottom + 4,
                                        self.rect.width, min(200, 28 * len(self.options)))
                if drop_rect.collidepoint(event.pos):
                    idx = (event.pos[1] - drop_rect.y - 4) // 28
                    if 0 <= idx < len(self.options):
                        self.selected_index = int(idx)
                    self.expanded = False
                    return True
                self.expanded = False
        return False


def _load_tk_photo(path, max_width=900, max_height=620):
    image = tk.PhotoImage(file=path)
    ratio = max(float(image.width()) / max_width, float(image.height()) / max_height, 1.0)
    factor = int(np.ceil(ratio))
    if factor > 1:
        image = image.subsample(factor, factor)
    return image


def run_tk_dashboard(args):
    _ensure_dir(args.output_dir)
    plots_dir = os.path.join(args.output_dir, 'plots')
    preview_dir = os.path.join(args.output_dir, 'preview')
    _ensure_dir(plots_dir)
    _ensure_dir(preview_dir)

    root = tk.Tk()
    root.title('RL-Frenet-CARLA Artifact Dashboard')
    root.geometry('1280x760')

    style = ttk.Style(root)
    try:
        style.theme_use('clam')
    except tk.TclError:
        pass

    state = {
        'agent_ids': _scan_agents(args.base_dir, args.agent_ids),
        'label_map': {},
        'selected_agents': {},
        'active_agent_id': None,
        'trajectory_files': [],
        'selected_traj_path': None,
        'auto_refresh': tk.BooleanVar(value=True),
        'last_action': None,
        'photo': None,
    }
    state['label_map'] = {agent_id: _find_agent_label(agent_id, args.base_dir)
                          for agent_id in state['agent_ids']}
    state['selected_agents'] = {agent_id: tk.BooleanVar(value=(agent_id in args.agent_ids))
                                for agent_id in state['agent_ids']}
    if state['agent_ids']:
        state['active_agent_id'] = args.traj_agent_id if args.traj_agent_id is not None else state['agent_ids'][0]

    root.columnconfigure(1, weight=1)
    root.rowconfigure(0, weight=1)
    side = ttk.Frame(root, padding=12)
    side.grid(row=0, column=0, sticky='ns')
    main_panel = ttk.Frame(root, padding=12)
    main_panel.grid(row=0, column=1, sticky='nsew')
    main_panel.rowconfigure(1, weight=1)
    main_panel.columnconfigure(0, weight=1)

    ttk.Label(side, text='RL-Frenet-CARLA', font=('Segoe UI', 16, 'bold')).pack(anchor='w')
    ttk.Label(side, text='Non-SDL dashboard for parallel visualizer use').pack(anchor='w', pady=(0, 12))

    agent_var = tk.StringVar()
    traj_var = tk.StringVar()
    status_var = tk.StringVar(value='Ready')
    summary_var = tk.StringVar(value='')

    ttk.Label(side, text='Active agent').pack(anchor='w')
    agent_combo = ttk.Combobox(side, textvariable=agent_var, width=34, state='readonly')
    agent_combo.pack(fill='x', pady=(2, 10))

    ttk.Label(side, text='Trajectory file').pack(anchor='w')
    traj_combo = ttk.Combobox(side, textvariable=traj_var, width=34, state='readonly')
    traj_combo.pack(fill='x', pady=(2, 10))

    image_label = ttk.Label(main_panel, anchor='center')
    image_label.grid(row=1, column=0, sticky='nsew')
    title_label = ttk.Label(main_panel, text='Training Artifact Viewer', font=('Segoe UI', 14, 'bold'))
    title_label.grid(row=0, column=0, sticky='w', pady=(0, 8))
    status_label = ttk.Label(main_panel, textvariable=status_var, wraplength=900)
    status_label.grid(row=2, column=0, sticky='ew', pady=(8, 0))
    summary_label = ttk.Label(side, textvariable=summary_var, justify='left')
    summary_label.pack(anchor='w', fill='x', pady=(4, 12))

    checks_frame = ttk.LabelFrame(side, text='Agents for comparison', padding=8)
    checks_frame.pack(fill='both', expand=True, pady=(10, 10))

    def selected_agent_ids():
        return [agent_id for agent_id, var in state['selected_agents'].items() if var.get()]

    def set_image(path):
        state['photo'] = _load_tk_photo(path)
        image_label.configure(image=state['photo'])

    def update_summary():
        agent_id = state['active_agent_id']
        if agent_id is None:
            summary_var.set('No agent found.')
            return
        artifact = _agent_summary(agent_id, args.base_dir)
        traj = _trajectory_summary(state['selected_traj_path'])
        lines = [
            'models: {}'.format(artifact.get('models', 0)),
            'ego csv: {}'.format(artifact.get('trajectory_files', 0)),
            'plan csv: {}'.format(artifact.get('plan_files', 0)),
            'traffic csv: {}'.format(artifact.get('surrounding_files', 0)),
            'episodes: {}'.format(artifact.get('episodes', 0)),
            'timesteps: {}'.format(artifact.get('timesteps', 0)),
        ]
        if traj.get('kind') == 'planner':
            lines.append('planner candidates: {}'.format(traj.get('candidate_paths', 0)))
            lines.append('selected points: {}'.format(traj.get('selected_points', 0)))
        elif traj.get('kind') == 'ego':
            lines.append('ego samples: {}'.format(traj.get('samples', 0)))
            lines.append('mean speed: {:.2f} m/s'.format(traj.get('mean_speed', 0.0)))
        elif traj.get('kind') == 'surrounding':
            lines.append('traffic actors: {}'.format(traj.get('traffic_actors', 0)))
            lines.append('traffic samples: {}'.format(traj.get('traffic_samples', 0)))
        summary_var.set('\n'.join(lines))

    def rebuild_agent_checks():
        for child in checks_frame.winfo_children():
            child.destroy()
        for agent_id in state['agent_ids']:
            label = state['label_map'].get(agent_id, 'agent_{}'.format(agent_id))
            var = state['selected_agents'].setdefault(agent_id, tk.BooleanVar(value=False))
            ttk.Checkbutton(checks_frame, text=label, variable=var).pack(anchor='w')

    def refresh_sources():
        current = state['active_agent_id']
        state['agent_ids'] = _scan_agents(args.base_dir, state['agent_ids'])
        state['label_map'] = {agent_id: _find_agent_label(agent_id, args.base_dir)
                              for agent_id in state['agent_ids']}
        for agent_id in state['agent_ids']:
            state['selected_agents'].setdefault(agent_id, tk.BooleanVar(value=(agent_id in args.agent_ids)))
        agent_values = ['{} (agent_{})'.format(state['label_map'][a].split(' (')[0], a)
                        for a in state['agent_ids']]
        agent_combo.configure(values=agent_values)
        if state['agent_ids']:
            if current in state['agent_ids']:
                idx = state['agent_ids'].index(current)
            else:
                idx = 0
            agent_combo.current(idx)
            state['active_agent_id'] = state['agent_ids'][idx]
        else:
            state['active_agent_id'] = None

        agent_id = state['active_agent_id']
        state['trajectory_files'] = _list_trajectory_files(agent_id, args.base_dir) if agent_id is not None else []
        traj_combo.configure(values=[_short_file_label(path) for path in state['trajectory_files']])
        if state['trajectory_files']:
            traj_combo.current(0)
            state['selected_traj_path'] = state['trajectory_files'][0]
        else:
            traj_var.set('')
            state['selected_traj_path'] = None
        rebuild_agent_checks()
        update_summary()

    def on_agent_changed(event=None):
        if not state['agent_ids']:
            return
        idx = agent_combo.current()
        if idx < 0:
            return
        state['active_agent_id'] = state['agent_ids'][idx]
        state['trajectory_files'] = _list_trajectory_files(state['active_agent_id'], args.base_dir)
        traj_combo.configure(values=[_short_file_label(path) for path in state['trajectory_files']])
        if state['trajectory_files']:
            traj_combo.current(0)
            state['selected_traj_path'] = state['trajectory_files'][0]
        else:
            traj_var.set('')
            state['selected_traj_path'] = None
        update_summary()

    def on_traj_changed(event=None):
        idx = traj_combo.current()
        if 0 <= idx < len(state['trajectory_files']):
            state['selected_traj_path'] = state['trajectory_files'][idx]
        update_summary()

    def plot_rewards_action(save_to_plots=True):
        chosen = selected_agent_ids()
        if not chosen:
            status_var.set('Select at least one agent.')
            return
        labels = [state['label_map'][a] for a in chosen]
        output_path = os.path.join(
            plots_dir if save_to_plots else preview_dir,
            'rewards_{}.png'.format(int(time.time())) if save_to_plots else 'rewards_preview.png'
        )
        _plot_rewards(chosen, labels, args.window_size, args.base_dir, output_path)
        set_image(output_path)
        status_var.set(('Rewards plot saved: {}' if save_to_plots else 'Rewards preview refreshed: {}').format(output_path))
        state['last_action'] = plot_rewards_action

    def plot_trajectory_action(save_to_plots=True):
        traj_path = state['selected_traj_path'] or _latest_trajectory(state['active_agent_id'], args.base_dir)
        if traj_path is None:
            status_var.set('No trajectory files found for agent_{}.'.format(state['active_agent_id']))
            return
        output_path = os.path.join(
            plots_dir if save_to_plots else preview_dir,
            'trajectory_{}.png'.format(int(time.time())) if save_to_plots else 'trajectory_preview.png'
        )
        _plot_trajectory(traj_path, output_path)
        set_image(output_path)
        status_var.set(('Trajectory plot saved: {}' if save_to_plots else 'Trajectory preview refreshed: {}').format(output_path))
        state['last_action'] = plot_trajectory_action
        update_summary()

    def plot_surrounding_action(save_to_plots=True):
        track_path = state['selected_traj_path']
        if _trajectory_summary(track_path).get('kind') != 'surrounding':
            track_path = _latest_surrounding(state['active_agent_id'], args.base_dir)
        if track_path is None:
            status_var.set('No surrounding trajectory files found for agent_{}.'.format(state['active_agent_id']))
            return
        output_path = os.path.join(
            plots_dir if save_to_plots else preview_dir,
            'surrounding_tracks_{}.png'.format(int(time.time())) if save_to_plots else 'surrounding_tracks_preview.png'
        )
        _plot_surrounding_tracks(track_path, output_path)
        set_image(output_path)
        status_var.set(('Surrounding trajectory plot saved: {}' if save_to_plots else 'Surrounding trajectory preview refreshed: {}').format(output_path))
        state['last_action'] = plot_surrounding_action
        update_summary()

    def plot_comparison_action(save_to_plots=True):
        chosen = selected_agent_ids()
        if not chosen:
            status_var.set('Select at least one agent.')
            return
        labels = [state['label_map'][a] for a in chosen]
        output_path = os.path.join(
            plots_dir if save_to_plots else preview_dir,
            'comparison_{}.png'.format(int(time.time())) if save_to_plots else 'comparison_preview.png'
        )
        _plot_comparison(chosen, labels, args.window_size, args.base_dir, output_path)
        set_image(output_path)
        status_var.set(('Comparison plot saved: {}' if save_to_plots else 'Comparison preview refreshed: {}').format(output_path))
        state['last_action'] = plot_comparison_action

    def plot_overview_action(save_to_plots=True):
        chosen = selected_agent_ids()
        if not chosen:
            status_var.set('Select at least one agent.')
            return
        labels = [state['label_map'][a] for a in chosen]
        output_path = os.path.join(
            plots_dir if save_to_plots else preview_dir,
            'artifact_overview_{}.png'.format(int(time.time())) if save_to_plots else 'artifact_overview_preview.png'
        )
        _plot_agent_overview(chosen, labels, args.base_dir, output_path)
        set_image(output_path)
        status_var.set(('Artifact overview saved: {}' if save_to_plots else 'Artifact overview preview refreshed: {}').format(output_path))
        state['last_action'] = plot_overview_action

    def export_action():
        chosen = selected_agent_ids()
        if not chosen:
            status_var.set('Select at least one agent.')
            return
        export_root = _export_artifacts(chosen, args.base_dir, plots_dir, args.export_dir)
        status_var.set('Exported artifacts to {}'.format(export_root))

    def open_file_action():
        file_path = _ask_open_file(os.path.join(args.base_dir, 'agent_{}'.format(state['active_agent_id'])))
        if file_path:
            state['selected_traj_path'] = file_path
            status_var.set('Selected trajectory: {}'.format(os.path.basename(file_path)))
            update_summary()

    def previous_file_action():
        if not state['trajectory_files']:
            status_var.set('No trajectory files found.')
            return
        idx = traj_combo.current()
        idx = (idx - 1) % len(state['trajectory_files'])
        traj_combo.current(idx)
        on_traj_changed()
        plot_trajectory_action(save_to_plots=True)

    def next_file_action():
        if not state['trajectory_files']:
            status_var.set('No trajectory files found.')
            return
        idx = traj_combo.current()
        idx = (idx + 1) % len(state['trajectory_files'])
        traj_combo.current(idx)
        on_traj_changed()
        plot_trajectory_action(save_to_plots=True)

    for label, command in [
        ('Plot rewards', plot_rewards_action),
        ('Plot trajectory', plot_trajectory_action),
        ('Ego + traffic', plot_surrounding_action),
        ('Plot compare', plot_comparison_action),
        ('Artifact map', plot_overview_action),
        ('Prev file', previous_file_action),
        ('Next file', next_file_action),
        ('Open file', open_file_action),
        ('Refresh', refresh_sources),
        ('Export', export_action),
    ]:
        ttk.Button(side, text=label, command=command).pack(fill='x', pady=2)
    ttk.Checkbutton(side, text='Auto refresh', variable=state['auto_refresh']).pack(anchor='w', pady=(8, 0))

    def auto_refresh_tick():
        if state['auto_refresh'].get():
            refresh_sources()
            if state['last_action'] is not None:
                state['last_action'](save_to_plots=False)
        root.after(10000, auto_refresh_tick)

    agent_combo.bind('<<ComboboxSelected>>', on_agent_changed)
    traj_combo.bind('<<ComboboxSelected>>', on_traj_changed)
    root.bind('<Left>', lambda event: previous_file_action())
    root.bind('<Right>', lambda event: next_file_action())
    root.bind('r', lambda event: plot_rewards_action())
    root.bind('t', lambda event: plot_trajectory_action())
    root.bind('s', lambda event: plot_surrounding_action())
    root.bind('o', lambda event: plot_overview_action())

    refresh_sources()
    status_var.set('Ready. This Tk backend does not create a pygame/SDL window.')
    root.after(10000, auto_refresh_tick)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description='Pygame UI for RL visualization.')
    parser.add_argument('--agent_ids', nargs='+', type=int, default=[1])
    parser.add_argument('--traj_agent_id', type=int, default=None)
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--base_dir', type=str, default='logs')
    parser.add_argument('--output_dir', type=str, default='outputs/pygame_ui')
    parser.add_argument('--export_dir', type=str, default='outputs/exports')
    parser.add_argument('--backend', choices=['pygame', 'tk'], default='pygame',
                        help='Use tk when running beside the CARLA pygame visualizer.')
    args = parser.parse_args()

    if args.backend == 'tk':
        run_tk_dashboard(args)
        return

    global pygame
    try:
        import pygame as pygame_module
        pygame = pygame_module
    except ImportError:
        raise RuntimeError('pygame is not installed. Use --backend tk or install pygame.')

    _ensure_dir(args.output_dir)
    plots_dir = os.path.join(args.output_dir, 'plots')
    preview_dir = os.path.join(args.output_dir, 'preview')
    _ensure_dir(plots_dir)
    _ensure_dir(preview_dir)

    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption('RL Visualization Dashboard')

    font = pygame.font.SysFont('consolas', 18)
    font_title = pygame.font.SysFont('consolas', 24, bold=True)

    agent_ids = _scan_agents(args.base_dir, args.agent_ids)
    label_map = {agent_id: _find_agent_label(agent_id, args.base_dir) for agent_id in agent_ids}
    selected_agents = {agent_id: (agent_id in args.agent_ids) for agent_id in agent_ids}
    traj_agent_id = args.traj_agent_id if args.traj_agent_id is not None else (agent_ids[0] if agent_ids else None)
    active_agent_id = traj_agent_id

    agent_options = ['{} (agent_{})'.format(label_map[a].split(' (')[0], a) for a in agent_ids]
    agent_dropdown = Dropdown((18, 164, 284, 30), agent_options, 0)

    trajectory_files = _list_trajectory_files(active_agent_id, args.base_dir) if active_agent_id is not None else []
    traj_labels = [_short_file_label(path) for path in trajectory_files]
    traj_dropdown = Dropdown((18, 244, 284, 30), traj_labels, 0)
    selected_traj_path = trajectory_files[0] if trajectory_files else None

    last_action = None
    auto_refresh = True
    refresh_interval = 10.0
    next_refresh = time.time() + refresh_interval

    status = 'Ready'
    image_surface = None
    image_rect = None
    current_image_path = None
    artifact_summary = _agent_summary(active_agent_id, args.base_dir) if active_agent_id is not None else {}
    trajectory_meta = _trajectory_summary(selected_traj_path)

    def refresh_sources():
        nonlocal agent_ids, label_map, selected_agents, traj_agent_id, active_agent_id
        nonlocal agent_options, trajectory_files, traj_labels, selected_traj_path
        nonlocal artifact_summary, trajectory_meta
        agent_ids = _scan_agents(args.base_dir, agent_ids)
        label_map = {agent_id: _find_agent_label(agent_id, args.base_dir) for agent_id in agent_ids}
        selected_agents = {agent_id: selected_agents.get(agent_id, False) for agent_id in agent_ids}
        if not any(selected_agents.values()) and agent_ids:
            selected_agents[agent_ids[0]] = True

        agent_options = ['{} (agent_{})'.format(label_map[a].split(' (')[0], a) for a in agent_ids]
        agent_dropdown.set_options(agent_options)

        if agent_ids:
            active_agent_id = agent_ids[agent_dropdown.selected_index]
            traj_agent_id = active_agent_id

        trajectory_files = _list_trajectory_files(active_agent_id, args.base_dir) if active_agent_id is not None else []
        traj_labels = [_short_file_label(path) for path in trajectory_files]
        traj_dropdown.set_options(traj_labels)
        selected_traj_path = trajectory_files[traj_dropdown.selected_index] if trajectory_files else None
        artifact_summary = _agent_summary(active_agent_id, args.base_dir) if active_agent_id is not None else {}
        trajectory_meta = _trajectory_summary(selected_traj_path)

    def plot_rewards_action(save_to_plots=True):
        nonlocal image_surface, image_rect, status, last_action, current_image_path
        chosen = [agent_id for agent_id, enabled in selected_agents.items() if enabled]
        if not chosen:
            status = 'Select at least one agent.'
            return
        labels = [label_map[a] for a in chosen]
        output_path = os.path.join(
            plots_dir if save_to_plots else preview_dir,
            'rewards_{}.png'.format(int(time.time())) if save_to_plots else 'rewards_preview.png'
        )
        _plot_rewards(chosen, labels, args.window_size, args.base_dir, output_path)
        image_surface, image_rect = _load_image(output_path,
                                               pygame.Rect(PANEL_WIDTH + 10, 10,
                                                           WINDOW_SIZE[0] - PANEL_WIDTH - 20,
                                                           WINDOW_SIZE[1] - 20))
        current_image_path = output_path
        status = ('Rewards plot saved: {}' if save_to_plots else 'Rewards preview refreshed: {}').format(output_path)
        last_action = plot_rewards_action

    def plot_trajectory_action(save_to_plots=True):
        nonlocal image_surface, image_rect, status, last_action, current_image_path, trajectory_meta
        traj_path = selected_traj_path or _latest_trajectory(active_agent_id, args.base_dir)
        if traj_path is None:
            status = 'No trajectory files found for agent_{}'.format(active_agent_id)
            return
        output_path = os.path.join(
            plots_dir if save_to_plots else preview_dir,
            'trajectory_{}.png'.format(int(time.time())) if save_to_plots else 'trajectory_preview.png'
        )
        _plot_trajectory(traj_path, output_path)
        image_surface, image_rect = _load_image(output_path,
                                               pygame.Rect(PANEL_WIDTH + 10, 10,
                                                           WINDOW_SIZE[0] - PANEL_WIDTH - 20,
                                                           WINDOW_SIZE[1] - 20))
        current_image_path = output_path
        trajectory_meta = _trajectory_summary(traj_path)
        status = ('Trajectory plot saved: {}' if save_to_plots else 'Trajectory preview refreshed: {}').format(output_path)
        last_action = plot_trajectory_action

    def plot_surrounding_action(save_to_plots=True):
        nonlocal image_surface, image_rect, status, last_action, current_image_path, trajectory_meta
        track_path = selected_traj_path
        if _trajectory_summary(track_path).get('kind') != 'surrounding':
            track_path = _latest_surrounding(active_agent_id, args.base_dir)
        if track_path is None:
            status = 'No surrounding trajectory files found for agent_{}'.format(active_agent_id)
            return
        output_path = os.path.join(
            plots_dir if save_to_plots else preview_dir,
            'surrounding_tracks_{}.png'.format(int(time.time())) if save_to_plots else 'surrounding_tracks_preview.png'
        )
        _plot_surrounding_tracks(track_path, output_path)
        image_surface, image_rect = _load_image(output_path,
                                               pygame.Rect(PANEL_WIDTH + 10, 10,
                                                           WINDOW_SIZE[0] - PANEL_WIDTH - 20,
                                                           WINDOW_SIZE[1] - 20))
        current_image_path = output_path
        trajectory_meta = _trajectory_summary(track_path)
        status = ('Surrounding trajectory plot saved: {}' if save_to_plots else 'Surrounding trajectory preview refreshed: {}').format(output_path)
        last_action = plot_surrounding_action

    def plot_comparison_action(save_to_plots=True):
        nonlocal image_surface, image_rect, status, last_action, current_image_path
        chosen = [agent_id for agent_id, enabled in selected_agents.items() if enabled]
        if not chosen:
            status = 'Select at least one agent.'
            return
        labels = [label_map[a] for a in chosen]
        output_path = os.path.join(
            plots_dir if save_to_plots else preview_dir,
            'comparison_{}.png'.format(int(time.time())) if save_to_plots else 'comparison_preview.png'
        )
        _plot_comparison(chosen, labels, args.window_size, args.base_dir, output_path)
        image_surface, image_rect = _load_image(output_path,
                                               pygame.Rect(PANEL_WIDTH + 10, 10,
                                                           WINDOW_SIZE[0] - PANEL_WIDTH - 20,
                                                           WINDOW_SIZE[1] - 20))
        current_image_path = output_path
        status = ('Comparison plot saved: {}' if save_to_plots else 'Comparison preview refreshed: {}').format(output_path)
        last_action = plot_comparison_action

    def plot_overview_action(save_to_plots=True):
        nonlocal image_surface, image_rect, status, last_action, current_image_path
        chosen = [agent_id for agent_id, enabled in selected_agents.items() if enabled]
        if not chosen:
            status = 'Select at least one agent.'
            return
        labels = [label_map[a] for a in chosen]
        output_path = os.path.join(
            plots_dir if save_to_plots else preview_dir,
            'artifact_overview_{}.png'.format(int(time.time())) if save_to_plots else 'artifact_overview_preview.png'
        )
        _plot_agent_overview(chosen, labels, args.base_dir, output_path)
        image_surface, image_rect = _load_image(output_path,
                                               pygame.Rect(PANEL_WIDTH + 10, 10,
                                                           WINDOW_SIZE[0] - PANEL_WIDTH - 20,
                                                           WINDOW_SIZE[1] - 20))
        current_image_path = output_path
        status = ('Artifact overview saved: {}' if save_to_plots else 'Artifact overview preview refreshed: {}').format(output_path)
        last_action = plot_overview_action

    def export_action():
        nonlocal status
        chosen = [agent_id for agent_id, enabled in selected_agents.items() if enabled]
        if not chosen:
            status = 'Select at least one agent.'
            return
        export_root = _export_artifacts(chosen, args.base_dir, plots_dir, args.export_dir)
        status = 'Exported artifacts to {}'.format(export_root)

    def open_file_action():
        nonlocal selected_traj_path, status, last_action, trajectory_meta
        file_path = _ask_open_file(os.path.join(args.base_dir, 'agent_{}'.format(active_agent_id)))
        if file_path:
            selected_traj_path = file_path
            trajectory_meta = _trajectory_summary(selected_traj_path)
            status = 'Selected trajectory: {}'.format(os.path.basename(file_path))
            last_action = plot_trajectory_action
        else:
            status = 'No file selected.'

    def refresh_action():
        nonlocal status
        refresh_sources()
        status = 'Lists refreshed.'

    def auto_refresh_action():
        nonlocal auto_refresh, status
        auto_refresh = not auto_refresh
        status = 'Auto refresh {}'.format('enabled' if auto_refresh else 'disabled')

    def previous_file_action():
        nonlocal selected_traj_path, status, trajectory_meta
        if not trajectory_files:
            status = 'No trajectory files found.'
            return
        traj_dropdown.selected_index = (traj_dropdown.selected_index - 1) % len(trajectory_files)
        selected_traj_path = trajectory_files[traj_dropdown.selected_index]
        trajectory_meta = _trajectory_summary(selected_traj_path)
        plot_trajectory_action(save_to_plots=True)

    def next_file_action():
        nonlocal selected_traj_path, status, trajectory_meta
        if not trajectory_files:
            status = 'No trajectory files found.'
            return
        traj_dropdown.selected_index = (traj_dropdown.selected_index + 1) % len(trajectory_files)
        selected_traj_path = trajectory_files[traj_dropdown.selected_index]
        trajectory_meta = _trajectory_summary(selected_traj_path)
        plot_trajectory_action(save_to_plots=True)

    buttons = [
        Button((18, 300, 136, 30), 'Plot rewards', plot_rewards_action),
        Button((166, 300, 136, 30), 'Plot trajectory', plot_trajectory_action),
        Button((18, 338, 136, 30), 'Ego + traffic', plot_surrounding_action),
        Button((166, 338, 136, 30), 'Plot compare', plot_comparison_action),
        Button((18, 376, 136, 30), 'Artifact map', plot_overview_action),
        Button((166, 376, 136, 30), 'Open file', open_file_action),
        Button((18, 414, 136, 30), 'Refresh', refresh_action),
        Button((166, 414, 136, 30), 'Prev file', previous_file_action),
        Button((18, 452, 136, 30), 'Next file', next_file_action),
        Button((166, 452, 136, 30), 'Export', export_action),
        Button((18, 490, 284, 30), 'Auto: ON', auto_refresh_action)
    ]

    running = True
    while running:
        if auto_refresh and last_action and time.time() >= next_refresh:
            last_action(save_to_plots=False)
            next_refresh = time.time() + refresh_interval

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_LEFT:
                    previous_file_action()
                elif event.key == pygame.K_RIGHT:
                    next_file_action()
                elif event.key == pygame.K_r:
                    plot_rewards_action()
                elif event.key == pygame.K_t:
                    plot_trajectory_action()
                elif event.key == pygame.K_s:
                    plot_surrounding_action()
                elif event.key == pygame.K_o:
                    plot_overview_action()

            handled = False
            if agent_dropdown.handle_event(event):
                handled = True
            if traj_dropdown.handle_event(event):
                handled = True

            if not handled:
                for button in buttons:
                    button.handle_event(event)

                if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    agent_list_top = 558
                    item_height = 22
                    list_bottom = WINDOW_SIZE[1] - 24
                    max_items = max(0, (list_bottom - agent_list_top) // item_height)
                    for idx, agent_id in enumerate(agent_ids[:max_items]):
                        rect = pygame.Rect(18, agent_list_top + idx * item_height, 284, item_height)
                        if rect.collidepoint(event.pos):
                            selected_agents[agent_id] = not selected_agents.get(agent_id, False)
                            break

        if agent_dropdown.options:
            active_agent_id = agent_ids[agent_dropdown.selected_index]
            trajectory_files = _list_trajectory_files(active_agent_id, args.base_dir)
            traj_labels = [_short_file_label(path) for path in trajectory_files]
            traj_dropdown.set_options(traj_labels)
            if traj_dropdown.selected_index < len(trajectory_files):
                selected_traj_path = trajectory_files[traj_dropdown.selected_index]
            else:
                selected_traj_path = trajectory_files[0] if trajectory_files else None
            artifact_summary = _agent_summary(active_agent_id, args.base_dir)
            trajectory_meta = _trajectory_summary(selected_traj_path)

        buttons[-1].label = 'Auto: {}'.format('ON' if auto_refresh else 'OFF')

        screen.fill(BACKGROUND)
        pygame.draw.rect(screen, PANEL_BG, pygame.Rect(0, 0, PANEL_WIDTH, WINDOW_SIZE[1]))

        _draw_text(screen, 'RL Dashboard', font_title, TEXT_MAIN, 18, 18)
        _draw_text(screen, 'Window: {}'.format(args.window_size), font, TEXT_MUTED, 18, 56)
        _draw_text(screen, 'Models: {}  Ego CSV: {}  Plan CSV: {}  Traffic CSV: {}'.format(
            artifact_summary.get('models', 0),
            artifact_summary.get('trajectory_files', 0),
            artifact_summary.get('plan_files', 0),
            artifact_summary.get('surrounding_files', 0)), font, TEXT_MUTED, 18, 80)
        if trajectory_meta.get('kind') == 'planner':
            _draw_text(screen, 'Planner: {} candidates, {} selected pts'.format(
                trajectory_meta.get('candidate_paths', 0),
                trajectory_meta.get('selected_points', 0)), font, WARNING, 18, 104)
        elif trajectory_meta.get('kind') == 'ego':
            _draw_text(screen, 'Ego trace: {} samples, {:.2f} m/s mean'.format(
                trajectory_meta.get('samples', 0),
                trajectory_meta.get('mean_speed', 0.0)), font, CHECK_ON, 18, 104)
        elif trajectory_meta.get('kind') == 'surrounding':
            _draw_text(screen, 'Traffic: {} actors, {} samples'.format(
                trajectory_meta.get('traffic_actors', 0),
                trajectory_meta.get('traffic_samples', 0)), font, WARNING, 18, 104)
        else:
            _draw_text(screen, 'No trajectory artifact selected', font, TEXT_MUTED, 18, 104)
        _draw_text(screen, 'Active agent', font, TEXT_MUTED, 18, 136)
        agent_dropdown.draw(screen, font)
        _draw_text(screen, 'Trajectory file', font, TEXT_MUTED, 18, 216)
        traj_dropdown.draw(screen, font)

        for button in buttons:
            button.draw(screen, font)

        _draw_text(screen, 'Agents for comparison', font, TEXT_MUTED, 18, 534)
        agent_list_top = 558
        item_height = 22
        list_bottom = WINDOW_SIZE[1] - 24
        max_items = max(0, (list_bottom - agent_list_top) // item_height)
        for idx, agent_id in enumerate(agent_ids[:max_items]):
            rect = pygame.Rect(18, agent_list_top + idx * item_height, 284, item_height)
            pygame.draw.rect(screen, PANEL_BG, rect)
            checkbox = pygame.Rect(rect.x + 2, rect.y + 3, 16, 16)
            pygame.draw.rect(screen, BUTTON_BORDER, checkbox, width=1)
            if selected_agents.get(agent_id, False):
                pygame.draw.rect(screen, CHECK_ON, checkbox.inflate(-4, -4))
            label = label_map.get(agent_id, 'agent_{}'.format(agent_id))
            _draw_text(screen, label, font, TEXT_MAIN, rect.x + 26, rect.y + 2)

        status_rect = pygame.Rect(PANEL_WIDTH + 16, WINDOW_SIZE[1] - 76,
                                  WINDOW_SIZE[0] - PANEL_WIDTH - 32, 58)
        pygame.draw.rect(screen, BUTTON_BG, status_rect, border_radius=8)
        pygame.draw.rect(screen, BUTTON_BORDER, status_rect, width=1, border_radius=8)
        _draw_text(screen, 'Output: {}'.format(args.output_dir), font, TEXT_MUTED,
                   status_rect.x + 12, status_rect.y + 8)
        _draw_wrapped_text(screen, 'Status: {}'.format(status), font, TEXT_MAIN,
                           pygame.Rect(status_rect.x + 12, status_rect.y + 30,
                                       status_rect.width - 24, 24))

        if image_surface is not None and image_rect is not None:
            screen.blit(image_surface, image_rect)
        else:
            hero_rect = pygame.Rect(PANEL_WIDTH + 24, 36,
                                    WINDOW_SIZE[0] - PANEL_WIDTH - 48,
                                    WINDOW_SIZE[1] - 140)
            pygame.draw.rect(screen, (24, 29, 36), hero_rect, border_radius=8)
            pygame.draw.rect(screen, BUTTON_BORDER, hero_rect, width=1, border_radius=8)
            _draw_text(screen, 'RL-Frenet-CARLA Training Artifact Browser',
                       font_title, TEXT_MAIN, hero_rect.x + 30, hero_rect.y + 30)
            _draw_text(screen, 'Use rewards, trajectory, comparison, or artifact map to generate evidence images.',
                       font, TEXT_MUTED, hero_rect.x + 30, hero_rect.y + 70)
            _draw_text(screen, 'Planner CSV files show multiple candidate trajectories and the selected trajectory.',
                       font, TEXT_MUTED, hero_rect.x + 30, hero_rect.y + 100)

        agent_dropdown.draw_overlay(screen, font)
        traj_dropdown.draw_overlay(screen, font)

        pygame.display.flip()

    pygame.quit()


if __name__ == '__main__':
    main()
