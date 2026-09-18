import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.animation as animation
matplotlib.use('Agg')


def render_refinement_video(
    frames,
    video_path="refinement.mp4",
    fps=15,
    preplaced_mask=None,   # Replaced fixed_mask with preplaced_mask
    fixed_size_mask=None,  # Added mask for [R] label
    pins=None,
    constraints=None,
    lr=None,
    loss=None,
):
    """
    Renders the layout optimization (refinement) process video.
    Supports visualization of clusters, MIB, and boundary blocks.
    """
    num_frames = len(frames)
    if num_frames == 0:
        return

    # Protect against missing directory
    output_dir = os.path.dirname(video_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if preplaced_mask is None:
        preplaced_mask = [False] * len(frames[0])
    if fixed_size_mask is None:
        fixed_size_mask = [False] * len(frames[0])

    # Find global boundaries across all frames
    all_rects = np.concatenate(frames, axis=0)
    max_x = np.max(all_rects[:, 0] + all_rects[:, 2]) if len(all_rects) > 0 else 1
    max_y = np.max(all_rects[:, 1] + all_rects[:, 3]) if len(all_rects) > 0 else 1

    if pins is not None and len(pins) > 0:
        max_x = max(max_x, np.max(pins[:, 0]))
        max_y = max(max_y, np.max(pins[:, 1]))

    margin = 0.05
    fig, ax = plt.subplots(figsize=(10, 10))  # Increased size for readability
    ax.set_aspect('equal')
    ax.set_xlim([0, max_x * (1 + margin)])
    ax.set_ylim([0, max_y * (1 + margin)])
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    # Lists of graphic objects to update in the animation
    rect_patches = []
    text_patches = []
    boundary_artists = []
    cluster_artists = []

    hatch_patterns = ['///', '\\\\\\', 'xxx', '...', '+++', 'OO', '**', '--', '||']

    # ==========================================
    # 1. PREPARATION OF STRUCTURES (Clusters and Boundaries)
    # ==========================================
    cluster_blocks = {}
    if constraints is not None:
        for i in range(len(frames[0])):
            cid = int(constraints[i, 3])
            if cid > 0:
                cluster_blocks.setdefault(cid, []).append(i)

    # Initialization of cluster objects (spider web)
    cluster_lines = {}
    cluster_centroids = {}
    if cluster_blocks:
        cluster_colors = cm.Dark2(np.linspace(0, 1, len(cluster_blocks)))
        for idx, (cid, b_indices) in enumerate(cluster_blocks.items()):
            if len(b_indices) > 1:
                color = cluster_colors[idx]
                lines = []
                # Create empty lines
                for _ in b_indices:
                    line, = ax.plot([], [], linestyle='--', color=color, alpha=0.7, linewidth=2, zorder=1)
                    lines.append(line)
                    cluster_artists.append(line)
                cluster_lines[cid] = lines

                # Create centroid (star) via plot for convenient coordinate updating
                centroid_pt, = ax.plot([], [], marker='*', color=color, markersize=15, markeredgecolor='black',
                                       linestyle='', zorder=2)
                cluster_centroids[cid] = centroid_pt
                cluster_artists.append(centroid_pt)

    boundary_dict = {}  # Dictionary to store block boundary lines

    # ==========================================
    # 2. INITIALIZATION OF BLOCKS
    # ==========================================
    for i, rect in enumerate(frames[0]):
        x, y, w, h = rect

        # Constraints
        mib_id = int(constraints[i, 2]) if constraints is not None else 0
        cluster_id = int(constraints[i, 3]) if constraints is not None else 0
        boundary_code = int(constraints[i, 4]) if constraints is not None else 0

        if preplaced_mask[i]:
            facecolor, edgecolor = 'lightpink', 'deeppink'
        else:
            facecolor, edgecolor = 'lightblue', 'blue'

        # MIB hatching (does not change over time, initialize immediately)
        current_hatch = hatch_patterns[mib_id % len(hatch_patterns)] if mib_id > 0 else None

        patch = patches.Rectangle((x, y), w, h, linewidth=1,
                                  edgecolor=edgecolor, facecolor=facecolor, alpha=0.5,
                                  hatch=current_hatch, zorder=3)
        ax.add_patch(patch)
        rect_patches.append(patch)

        # Boundaries
        if boundary_code > 0:
            b_lines = []
            if boundary_code & 1:
                l, = ax.plot([], [], color='red', linewidth=3, zorder=4)
                b_lines.append(('left', l))
            if boundary_code & 2:
                l, = ax.plot([], [], color='red', linewidth=3, zorder=4)
                b_lines.append(('right', l))
            if boundary_code & 4:
                l, = ax.plot([], [], color='red', linewidth=3, zorder=4)
                b_lines.append(('top', l))
            if boundary_code & 8:
                l, = ax.plot([], [], color='red', linewidth=3, zorder=4)
                b_lines.append(('bottom', l))
            boundary_dict[i] = b_lines
            for _, l in b_lines:
                boundary_artists.append(l)

        # Text
        label = f"{i}"
        if fixed_size_mask[i]:
            label += "\n[R]"  # Add strict size label
        if mib_id > 0 or cluster_id > 0:
            if mib_id > 0: label += f"\nM:{mib_id}"
            if cluster_id > 0: label += f"\nC:{cluster_id}"

        cx = x + w / 2.0
        cy = y + h / 2.0
        txt = ax.text(cx, cy, label, ha='center', va='center', fontsize=7, color='black', weight='bold', zorder=5)
        text_patches.append(txt)

    # Pins
    if pins is not None and len(pins) > 0:
        ax.scatter(pins[:, 0], pins[:, 1], color='green', s=40, zorder=6, marker='o', label='Pins')

    title = ax.set_title("Refinement Step: 0")

    # ==========================================
    # 3. UPDATE FUNCTION (Called every frame)
    # ==========================================
    def update(frame_idx):
        rects = frames[frame_idx]

        # Recalculate cluster centroids
        current_centroids = {}
        if cluster_blocks:
            for cid, b_indices in cluster_blocks.items():
                if len(b_indices) > 1:
                    pts = np.array(
                        [[rects[bi][0] + rects[bi][2] / 2.0, rects[bi][1] + rects[bi][3] / 2.0] for bi in b_indices])
                    centroid = pts.mean(axis=0)
                    current_centroids[cid] = (centroid, pts)

        for i, rect in enumerate(rects):
            x, y, w, h = rect

            # Update rectangle
            rect_patches[i].set_xy((x, y))
            rect_patches[i].set_width(w)
            rect_patches[i].set_height(h)

            # Update text
            cx = x + w / 2.0
            cy = y + h / 2.0
            text_patches[i].set_position((cx, cy))

            # Update red lines (Boundaries)
            if i in boundary_dict:
                for b_type, l in boundary_dict[i]:
                    if b_type == 'left':
                        l.set_data([x, x], [y, y + h])
                    elif b_type == 'right':
                        l.set_data([x + w, x + w], [y, y + h])
                    elif b_type == 'top':
                        l.set_data([x, x + w], [y + h, y + h])
                    elif b_type == 'bottom':
                        l.set_data([x, x + w], [y, y])

        # Update cluster connections
        if cluster_blocks:
            for cid, (centroid, pts) in current_centroids.items():
                # Update star position
                cluster_centroids[cid].set_data([centroid[0]], [centroid[1]])
                # Update lines from star to blocks
                lines = cluster_lines[cid]
                for pt_idx, pt in enumerate(pts):
                    lines[pt_idx].set_data([centroid[0], pt[0]], [centroid[1], pt[1]])

        title_str = f"Refinement Step: {frame_idx + 1} / {num_frames}"
        if lr is not None and frame_idx < len(lr):
            title_str += f" | LR: {lr[frame_idx]:.6f}"
        if loss is not None and frame_idx < len(loss):
            title_str += f" | Loss: {loss[frame_idx]:.6f}"
        title.set_text(title_str)

        # For optimization (blit=True) return the list of all changed objects
        return rect_patches + text_patches + boundary_artists + cluster_artists + [title]

    # ==========================================
    # 4. RENDER
    # ==========================================
    ani = animation.FuncAnimation(
        fig, update, frames=num_frames,
        interval=1000 / fps, blit=True
    )

    print(f"Rendering video ({num_frames} frames)...")
    if video_path.endswith('.gif'):
        ani.save(video_path, writer='pillow', fps=fps)
    else:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        ani.save(video_path, writer=writer)

    plt.close(fig)
    print(f"Video saved to: {video_path}")


def draw_layout(
    rects,
    test_id,
    fixed_mask=None,
    fixed_size_mask=None,
    pins=None,
    constraints=None,
    score_info="",
    output_dir="validation_images_full"
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if fixed_mask is None:
        fixed_mask = [False] * len(rects)

    # Initialize fixed size mask if not provided
    if fixed_size_mask is None:
        fixed_size_mask = [False] * len(rects)

    max_x = np.max(rects[:, 0] + rects[:, 2]) if len(rects) > 0 else 1
    max_y = np.max(rects[:, 1] + rects[:, 3]) if len(rects) > 0 else 1

    if pins is not None and len(pins) > 0:
        max_x = max(max_x, np.max(pins[:, 0]))
        max_y = max(max_y, np.max(pins[:, 1]))

    margin = 0.05
    # Make the canvas slightly larger for easier reading of multi-line text
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')

    # ==========================================================
    # 1. RENDERING CLUSTER CONNECTIONS (Draw under blocks, zorder=1)
    # ==========================================================
    cluster_centers = {}
    if constraints is not None:
        for i, rect in enumerate(rects):
            cluster_id = int(constraints[i, 3])
            if cluster_id > 0:
                cx = rect[0] + rect[2] / 2.0
                cy = rect[1] + rect[3] / 2.0
                if cluster_id not in cluster_centers:
                    cluster_centers[cluster_id] = []
                cluster_centers[cluster_id].append((cx, cy))

    if cluster_centers:
        unique_clusters = list(cluster_centers.keys())
        # Get a contrasting palette for cluster connections
        cluster_colors = cm.Dark2(np.linspace(0, 1, len(unique_clusters)))

        for idx, cid in enumerate(unique_clusters):
            pts = np.array(cluster_centers[cid])
            if len(pts) > 1:
                # Find the center of mass of the cluster
                centroid = pts.mean(axis=0)
                color = cluster_colors[idx]
                # Draw spider web from centroid to blocks
                for pt in pts:
                    ax.plot([centroid[0], pt[0]], [centroid[1], pt[1]],
                            linestyle='--', color=color, alpha=0.7, linewidth=2, zorder=1)
                # Draw the centroid itself with a star
                ax.scatter(centroid[0], centroid[1], color=color, marker='*', s=150, zorder=2, edgecolors='black')

    # ==========================================================
    # 2. RENDERING BLOCKS (zorder=3)
    # ==========================================================
    hatch_patterns = ['///', '\\\\\\', 'xxx', '...', '+++', 'OO', '**', '--', '||']

    for i, rect in enumerate(rects):
        x, y, w, h = rect
        cx = x + w / 2.0
        cy = y + h / 2.0

        if fixed_mask[i]:
            facecolor, edgecolor = 'lightpink', 'deeppink'
        else:
            facecolor, edgecolor = 'lightblue', 'blue'

        # Extract constraints
        mib_id = 0
        cluster_id = 0
        boundary_code = 0
        if constraints is not None:
            mib_id = int(constraints[i, 2])
            cluster_id = int(constraints[i, 3])
            boundary_code = int(constraints[i, 4])

        # Apply hatch pattern for MIB
        current_hatch = None
        if mib_id > 0:
            current_hatch = hatch_patterns[mib_id % len(hatch_patterns)]

        patch = patches.Rectangle((x, y), w, h, linewidth=1,
                                  edgecolor=edgecolor, facecolor=facecolor, alpha=0.5,
                                  hatch=current_hatch, zorder=3)
        ax.add_patch(patch)

        # ----------------------------------------------------------
        # Rendering anchors for fixed size blocks (Fixed)
        # ----------------------------------------------------------
        if fixed_size_mask[i]:
            # To keep the squares perfectly shaped at any Aspect Ratio,
            # set their size as a fixed percentage (e.g., 8%) of the smaller side of the block
            s = min(w, h) * 0.08

            # Bottom left corner
            ax.add_patch(patches.Rectangle((x, y), s, s, facecolor='black', edgecolor='none', zorder=4.5))

            # Bottom right corner (shift left by the size of the square s itself)
            ax.add_patch(patches.Rectangle((x + w - s, y), s, s, facecolor='black', edgecolor='none', zorder=4.5))

            # Top left corner (shift down by the size of the square s itself)
            ax.add_patch(patches.Rectangle((x, y + h - s), s, s, facecolor='black', edgecolor='none', zorder=4.5))

            # Top right corner (shift both X and Y)
            ax.add_patch(patches.Rectangle((x + w - s, y + h - s), s, s, facecolor='black', edgecolor='none', zorder=4.5))

        if 0:
            if fixed_size_mask[i]:
                # Draw 4 black squares (marker 's' - square) at the corners of the block
                ax.plot([x, x + w, x + w, x], [y, y, y + h, y + h],
                        marker='s', color='black', markersize=4, linestyle='none', zorder=4.5)

        # ==========================================================
        # 3. RENDERING BOUNDARIES (On top of blocks, zorder=4)
        # ==========================================================
        if boundary_code > 0:
            b_color = 'red'
            b_lw = 3  # Thick red line
            # Bit flags: 1=Left, 2=Right, 4=Top, 8=Bottom
            if boundary_code & 1:  # Left
                ax.plot([x, x], [y, y + h], color=b_color, linewidth=b_lw, zorder=4)
            if boundary_code & 2:  # Right
                ax.plot([x + w, x + w], [y, y + h], color=b_color, linewidth=b_lw, zorder=4)
            if boundary_code & 4:  # Top
                ax.plot([x, x + w], [y + h, y + h], color=b_color, linewidth=b_lw, zorder=4)
            if boundary_code & 8:  # Bottom
                ax.plot([x, x + w], [y, y], color=b_color, linewidth=b_lw, zorder=4)

        # ==========================================================
        # 4. TEXT LABELS (zorder=5)
        # ==========================================================
        label = f"{i}"

        # Add rigid block indicator (Rigid)
        if fixed_size_mask[i]:
            label += "\n[R]"

        if mib_id > 0 or cluster_id > 0:
            if mib_id > 0: label += f"\nM:{mib_id}"
            if cluster_id > 0: label += f"\nC:{cluster_id}"

        ax.text(cx, cy, label, ha='center', va='center', fontsize=7, color='black', weight='bold', zorder=5)

    # ==========================================================
    # 5. RENDERING PINS (zorder=6)
    # ==========================================================
    if pins is not None and len(pins) > 0:
        ax.scatter(pins[:, 0], pins[:, 1],
                   color='green', s=40, zorder=6,
                   marker='o', label='Pins')

    ax.set_xlim([0, max_x * (1 + margin)])
    ax.set_ylim([0, max_y * (1 + margin)])
    ax.set_title(f"Test Case {test_id} Layout{score_info}")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    file_path = os.path.join(output_dir, f"test_{test_id}.png")
    plt.savefig(file_path, bbox_inches='tight')
    plt.close(fig)
    plt.close('all')