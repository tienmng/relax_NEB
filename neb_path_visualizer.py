#!/usr/bin/env python
"""
NEB Path Visualization Module
=============================
Creates crystal structure figures with overlaid oxygen atom positions
showing the migration path through NEB images.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from pymatgen.core import Structure
from pymatgen.io.vasp.inputs import Poscar
from typing import List, Dict, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)

class NEBPathVisualizer:
    """
    Visualizes NEB migration paths with crystal structures and oxygen hopping.
    """
    
    def __init__(self, atom_colors: Dict[str, str] = None, 
                 atom_sizes: Dict[str, float] = None):
        """
        Initialize visualizer with custom atom colors and sizes.
        
        Args:
            atom_colors: Dictionary mapping element symbols to colors
            atom_sizes: Dictionary mapping element symbols to sizes
        """
        # Default atom colors
        self.atom_colors = atom_colors or {
            'O': '#FF0000',      # Red for regular oxygen
            'La': '#00CED1',     # Dark turquoise
            'Ni': '#A0A0A0',     # Gray
            'V': '#FFA500',      # Orange
            'Fe': '#B22222',     # Firebrick
            'Co': '#FF1493',     # Deep pink
            'Mn': '#9370DB',     # Medium purple
            'Ti': '#708090',     # Slate gray
            'Nb': '#4682B4',     # Steel blue
        }
        
        # Default atom sizes (in Angstroms)
        self.atom_sizes = atom_sizes or {
            'O': 0.6,
            'La': 1.0,
            'Ni': 0.7,
            'V': 0.7,
            'Fe': 0.7,
            'Co': 0.7,
            'Mn': 0.7,
            'Ti': 0.8,
            'Nb': 0.8,
        }
        
        # Special colors for oxygen migration
        self.o_initial_color = '#FFD700'  # Gold/yellow for initial
        self.o_final_color = '#FF6347'    # Tomato red for final
        self.o_transition_color = '#FFA500'  # Orange for transition
        self.o_transition_alpha = 0.6      # Transparency for transition states
    
    def load_neb_structures(self, neb_dir: str, n_images: int,
                           file_manager=None) -> List[Structure]:
        """
        Load all NEB structures from a directory.
        
        Args:
            neb_dir: Directory containing NEB images
            n_images: Number of intermediate images (excluding endpoints)
            file_manager: FileManager instance for cleaning CONTCAR files
            
        Returns:
            List of Structure objects
        """
        structures = []
        
        for i in range(n_images + 2):
            img_dir = os.path.join(neb_dir, f"{i:02d}")
            
            # Try CONTCAR first, then POSCAR
            for filename in ["CONTCAR", "POSCAR"]:
                filepath = os.path.join(img_dir, filename)
                
                if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
                    try:
                        # Clean CONTCAR if needed
                        if filename == "CONTCAR" and file_manager:
                            logger.debug(f"Cleaning {filename} for image {i:02d}")
                            file_manager.clean_contcar_elements(filepath)
                        
                        structure = Structure.from_file(filepath)
                        structures.append(structure)
                        logger.debug(f"Loaded structure from {filepath}")
                        break
                        
                    except Exception as e:
                        logger.warning(f"Error loading {filepath}: {e}")
                        continue
            else:
                logger.error(f"No valid structure found for image {i:02d}")
                return []
        
        logger.info(f"Loaded {len(structures)} NEB structures")
        return structures
    
    def plot_migration_path_2d(self, structures: List[Structure], 
                              moving_atom_idx: int,
                              projection_plane: str = 'xy',
                              output_path: str = None,
                              show_atoms: bool = True,
                              show_cell: bool = True,
                              title: str = "Oxygen Migration Path") -> str:
        """
        Create 2D projection of crystal structure with oxygen migration path.
        
        Args:
            structures: List of NEB structures
            moving_atom_idx: Index of moving oxygen atom
            projection_plane: Plane to project onto ('xy', 'xz', or 'yz')
            output_path: Output file path
            show_atoms: Whether to show all atoms
            show_cell: Whether to show unit cell
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        if not structures:
            logger.error("No structures provided")
            return None
        
        # Set up projection indices
        plane_indices = {
            'xy': (0, 1, 2),
            'xz': (0, 2, 1),
            'yz': (1, 2, 0)
        }
        x_idx, y_idx, z_idx = plane_indices.get(projection_plane, (0, 1, 2))
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot unit cell if requested
        if show_cell:
            self._plot_unit_cell_2d(ax, structures[0], x_idx, y_idx)
        
        # Plot all atoms if requested (using first structure as reference)
        if show_atoms:
            ref_struct = structures[0]
            for i, site in enumerate(ref_struct):
                if i != moving_atom_idx:  # Skip the moving atom
                    element = str(site.specie)
                    color = self.atom_colors.get(element, '#808080')
                    size = self.atom_sizes.get(element, 0.5) * 100
                    
                    coords = site.coords
                    ax.scatter(coords[x_idx], coords[y_idx], 
                             c=color, s=size, alpha=0.7,
                             edgecolors='black', linewidth=0.5)
        
        # Plot oxygen migration path
        o_positions = []
        for struct in structures:
            o_positions.append(struct[moving_atom_idx].coords)
        o_positions = np.array(o_positions)
        
        # Plot transition oxygen positions (intermediate images)
        for i in range(1, len(structures) - 1):
            x, y = o_positions[i, x_idx], o_positions[i, y_idx]
            
            # Transparent fill with solid outline
            circle = Circle((x, y), 
                          radius=self.atom_sizes['O'] * 0.5,
                          facecolor=self.o_transition_color,
                          edgecolor=self.o_transition_color,
                          alpha=self.o_transition_alpha,
                          linewidth=2)
            ax.add_patch(circle)
            
            # Add image number
            ax.text(x, y, str(i), 
                   ha='center', va='center', 
                   fontsize=8, fontweight='bold')
        
        # Plot initial oxygen position (solid yellow)
        x_init, y_init = o_positions[0, x_idx], o_positions[0, y_idx]
        circle_init = Circle((x_init, y_init),
                           radius=self.atom_sizes['O'] * 0.5,
                           facecolor=self.o_initial_color,
                           edgecolor='black',
                           linewidth=2)
        ax.add_patch(circle_init)
        ax.text(x_init, y_init - self.atom_sizes['O'] * 0.8, 'Initial',
               ha='center', va='top', fontsize=10, fontweight='bold')
        
        # Plot final oxygen position (solid red)
        x_final, y_final = o_positions[-1, x_idx], o_positions[-1, y_idx]
        circle_final = Circle((x_final, y_final),
                            radius=self.atom_sizes['O'] * 0.5,
                            facecolor=self.o_final_color,
                            edgecolor='black',
                            linewidth=2)
        ax.add_patch(circle_final)
        ax.text(x_final, y_final - self.atom_sizes['O'] * 0.8, 'Final',
               ha='center', va='top', fontsize=10, fontweight='bold')
        
        # Draw migration path with arrows
        for i in range(len(o_positions) - 1):
            x1, y1 = o_positions[i, x_idx], o_positions[i, y_idx]
            x2, y2 = o_positions[i+1, x_idx], o_positions[i+1, y_idx]
            
            # Apply minimum image convention if needed
            dx, dy = x2 - x1, y2 - y1
            
            # Draw arrow
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                       arrowprops=dict(arrowstyle='->', 
                                     color='black', 
                                     lw=1.5,
                                     alpha=0.7))
        
        # Set labels and title
        ax.set_xlabel(f'{projection_plane[0].upper()} (Å)', fontsize=12)
        ax.set_ylabel(f'{projection_plane[1].upper()} (Å)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        legend_elements = [
            mpatches.Patch(color=self.o_initial_color, label='Initial O position'),
            mpatches.Patch(color=self.o_final_color, label='Final O position'),
            mpatches.Patch(color=self.o_transition_color, 
                         alpha=self.o_transition_alpha,
                         label='Transition O positions')
        ]
        ax.legend(handles=legend_elements, loc='best', fontsize=10)
        
        # Save figure
        if output_path is None:
            output_path = f"neb_path_{projection_plane}.png"
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved 2D migration path plot to {output_path}")
        return output_path
    
    def plot_migration_path_3d(self, structures: List[Structure],
                              moving_atom_idx: int,
                              output_path: str = None,
                              show_atoms: bool = True,
                              show_cell: bool = True,
                              view_angles: Tuple[float, float] = (30, 45),
                              title: str = "Oxygen Migration Path 3D") -> str:
        """
        Create 3D visualization of crystal structure with oxygen migration path.
        
        Args:
            structures: List of NEB structures
            moving_atom_idx: Index of moving oxygen atom
            output_path: Output file path
            show_atoms: Whether to show all atoms
            show_cell: Whether to show unit cell
            view_angles: Tuple of (elevation, azimuth) viewing angles
            title: Plot title
            
        Returns:
            Path to saved figure
        """
        if not structures:
            logger.error("No structures provided")
            return None
        
        # Create 3D figure
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot unit cell if requested
        if show_cell:
            self._plot_unit_cell_3d(ax, structures[0])
        
        # Plot all atoms if requested (using first structure)
        if show_atoms:
            ref_struct = structures[0]
            for i, site in enumerate(ref_struct):
                if i != moving_atom_idx:
                    element = str(site.specie)
                    color = self.atom_colors.get(element, '#808080')
                    size = self.atom_sizes.get(element, 0.5) * 200
                    
                    coords = site.coords
                    ax.scatter(coords[0], coords[1], coords[2],
                             c=color, s=size, alpha=0.7,
                             edgecolors='black', linewidth=0.5)
        
        # Get oxygen positions
        o_positions = []
        for struct in structures:
            o_positions.append(struct[moving_atom_idx].coords)
        o_positions = np.array(o_positions)
        
        # Plot transition oxygen positions
        for i in range(1, len(structures) - 1):
            x, y, z = o_positions[i]
            ax.scatter(x, y, z,
                     c=self.o_transition_color,
                     s=self.atom_sizes['O'] * 300,
                     alpha=self.o_transition_alpha,
                     edgecolors=self.o_transition_color,
                     linewidth=2)
            ax.text(x, y, z, str(i),
                   fontsize=8, fontweight='bold')
        
        # Plot initial oxygen position
        x_i, y_i, z_i = o_positions[0]
        ax.scatter(x_i, y_i, z_i,
                 c=self.o_initial_color,
                 s=self.atom_sizes['O'] * 400,
                 edgecolors='black',
                 linewidth=2,
                 label='Initial O')
        
        # Plot final oxygen position
        x_f, y_f, z_f = o_positions[-1]
        ax.scatter(x_f, y_f, z_f,
                 c=self.o_final_color,
                 s=self.atom_sizes['O'] * 400,
                 edgecolors='black',
                 linewidth=2,
                 label='Final O')
        
        # Draw migration path
        for i in range(len(o_positions) - 1):
            ax.plot([o_positions[i, 0], o_positions[i+1, 0]],
                   [o_positions[i, 1], o_positions[i+1, 1]],
                   [o_positions[i, 2], o_positions[i+1, 2]],
                   'k-', linewidth=2, alpha=0.7)
        
        # Set labels and title
        ax.set_xlabel('X (Å)', fontsize=12)
        ax.set_ylabel('Y (Å)', fontsize=12)
        ax.set_zlabel('Z (Å)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set viewing angle
        ax.view_init(elev=view_angles[0], azim=view_angles[1])
        
        # Add legend
        ax.legend(loc='best', fontsize=10)
        
        # Save figure
        if output_path is None:
            output_path = "neb_path_3d.png"
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved 3D migration path plot to {output_path}")
        return output_path
    
    def plot_migration_sequence(self, structures: List[Structure],
                               moving_atom_idx: int,
                               projection_plane: str = 'xy',
                               output_path: str = None,
                               n_cols: int = 3,
                               show_path: bool = True) -> str:
        """
        Create a sequence of structure snapshots showing oxygen migration.
        
        Args:
            structures: List of NEB structures
            moving_atom_idx: Index of moving oxygen atom
            projection_plane: Plane to project onto
            output_path: Output file path
            n_cols: Number of columns in subplot grid
            show_path: Whether to show migration path in each frame
            
        Returns:
            Path to saved figure
        """
        n_images = len(structures)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        # Create figure with subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Flatten axes array for easier iteration
        axes_flat = axes.flatten()
        
        # Get projection indices
        plane_indices = {
            'xy': (0, 1),
            'xz': (0, 2),
            'yz': (1, 2)
        }
        x_idx, y_idx = plane_indices.get(projection_plane, (0, 1))
        
        # Get oxygen positions for path
        o_positions = []
        for struct in structures:
            o_positions.append(struct[moving_atom_idx].coords)
        o_positions = np.array(o_positions)
        
        # Plot each structure
        for img_idx in range(n_images):
            ax = axes_flat[img_idx]
            struct = structures[img_idx]
            
            # Plot unit cell
            self._plot_unit_cell_2d(ax, struct, x_idx, y_idx)
            
            # Plot all atoms except moving oxygen
            for i, site in enumerate(struct):
                if i != moving_atom_idx:
                    element = str(site.specie)
                    color = self.atom_colors.get(element, '#808080')
                    size = self.atom_sizes.get(element, 0.5) * 50
                    
                    coords = site.coords
                    ax.scatter(coords[x_idx], coords[y_idx],
                             c=color, s=size, alpha=0.7,
                             edgecolors='black', linewidth=0.5)
            
            # Plot migration path if requested
            if show_path:
                # Plot path as dashed line
                path_x = o_positions[:, x_idx]
                path_y = o_positions[:, y_idx]
                ax.plot(path_x, path_y, 'k--', alpha=0.3, linewidth=1)
                
                # Mark all oxygen positions with small circles
                for j in range(len(o_positions)):
                    if j != img_idx:
                        ax.scatter(o_positions[j, x_idx], o_positions[j, y_idx],
                                 c='gray', s=30, alpha=0.3)
            
            # Highlight current oxygen position
            current_o = struct[moving_atom_idx]
            x, y = current_o.coords[x_idx], current_o.coords[y_idx]
            
            if img_idx == 0:
                color = self.o_initial_color
                label = "Initial"
            elif img_idx == n_images - 1:
                color = self.o_final_color
                label = "Final"
            else:
                color = self.o_transition_color
                label = f"Image {img_idx}"
            
            # Draw oxygen with appropriate style
            if img_idx > 0 and img_idx < n_images - 1:
                # Transition state: transparent with outline
                circle = Circle((x, y),
                              radius=self.atom_sizes['O'] * 0.3,
                              facecolor=color,
                              edgecolor=color,
                              alpha=self.o_transition_alpha,
                              linewidth=2)
                ax.add_patch(circle)
            else:
                # Initial/final: solid
                ax.scatter(x, y, c=color, s=200,
                         edgecolors='black', linewidth=2)
            
            # Set title and labels
            ax.set_title(label, fontsize=12, fontweight='bold')
            ax.set_xlabel(f'{projection_plane[0].upper()} (Å)', fontsize=10)
            ax.set_ylabel(f'{projection_plane[1].upper()} (Å)', fontsize=10)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_images, len(axes_flat)):
            axes_flat[idx].axis('off')
        
        # Main title
        fig.suptitle('Oxygen Migration Sequence', fontsize=16, fontweight='bold')
        
        # Save figure
        if output_path is None:
            output_path = "neb_migration_sequence.png"
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved migration sequence plot to {output_path}")
        return output_path
    
    def _plot_unit_cell_2d(self, ax, structure: Structure, x_idx: int, y_idx: int):
        """Helper to plot 2D projection of unit cell."""
        lattice = structure.lattice.matrix
        
        # Define unit cell vertices in fractional coordinates
        vertices_frac = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom face
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # top face
        ])
        
        # Convert to Cartesian
        vertices_cart = np.dot(vertices_frac, lattice)
        
        # Define edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # bottom face
            (4, 5), (5, 6), (6, 7), (7, 4),  # top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # vertical edges
        ]
        
        # Plot edges
        for edge in edges:
            v1, v2 = vertices_cart[edge[0]], vertices_cart[edge[1]]
            ax.plot([v1[x_idx], v2[x_idx]], [v1[y_idx], v2[y_idx]],
                   'k-', alpha=0.3, linewidth=1)
    
    def _plot_unit_cell_3d(self, ax, structure: Structure):
        """Helper to plot 3D unit cell."""
        lattice = structure.lattice.matrix
        
        # Define unit cell vertices
        vertices_frac = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ])
        
        vertices_cart = np.dot(vertices_frac, lattice)
        
        # Define edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        # Plot edges
        for edge in edges:
            v1, v2 = vertices_cart[edge[0]], vertices_cart[edge[1]]
            ax.plot([v1[0], v2[0]], [v1[1], v2[1]], [v1[2], v2[2]],
                   'k-', alpha=0.3, linewidth=1)
    
    def create_all_visualizations(self, neb_dir: str, n_images: int,
                                 moving_atom_idx: int,
                                 output_dir: str = None,
                                 file_manager=None) -> Dict[str, str]:
        """
        Create all visualization types for the NEB path.
        
        Args:
            neb_dir: Directory containing NEB calculation
            n_images: Number of intermediate images
            moving_atom_idx: Index of moving oxygen atom
            output_dir: Output directory for figures
            file_manager: FileManager instance
            
        Returns:
            Dictionary mapping visualization type to file path
        """
        if output_dir is None:
            output_dir = os.path.join(neb_dir, "visualizations")
        os.makedirs(output_dir, exist_ok=True)
        
        # Load structures
        structures = self.load_neb_structures(neb_dir, n_images, file_manager)
        if not structures:
            logger.error("Failed to load NEB structures")
            return {}
        
        created_files = {}
        
        # Create 2D projections
        for plane in ['xy', 'xz', 'yz']:
            try:
                path = self.plot_migration_path_2d(
                    structures, moving_atom_idx,
                    projection_plane=plane,
                    output_path=os.path.join(output_dir, f"migration_path_{plane}.png"),
                    title=f"Oxygen Migration Path - {plane.upper()} Projection"
                )
                created_files[f'2d_{plane}'] = path
            except Exception as e:
                logger.error(f"Error creating {plane} projection: {e}")
        
        # Create 3D visualization
        try:
            path = self.plot_migration_path_3d(
                structures, moving_atom_idx,
                output_path=os.path.join(output_dir, "migration_path_3d.png"),
                view_angles=(30, 45)
            )
            created_files['3d'] = path
            
            # Create another 3D view
            path = self.plot_migration_path_3d(
                structures, moving_atom_idx,
                output_path=os.path.join(output_dir, "migration_path_3d_alt.png"),
                view_angles=(60, 120),
                title="Oxygen Migration Path 3D (Alternative View)"
            )
            created_files['3d_alt'] = path
        except Exception as e:
            logger.error(f"Error creating 3D visualization: {e}")
        
        # Create sequence visualization
        try:
            path = self.plot_migration_sequence(
                structures, moving_atom_idx,
                projection_plane='xy',
                output_path=os.path.join(output_dir, "migration_sequence.png"),
                n_cols=3
            )
            created_files['sequence'] = path
        except Exception as e:
            logger.error(f"Error creating sequence visualization: {e}")
        
        logger.info(f"Created {len(created_files)} visualizations in {output_dir}")
        return created_files


# Example usage function
def visualize_neb_path(neb_dir: str, n_images: int, moving_atom_idx: int,
                      output_dir: str = None, file_manager=None):
    """
    Convenience function to create all NEB visualizations.
    
    Args:
        neb_dir: NEB calculation directory (e.g., "./multistage_neb/neb_stage3")
        n_images: Number of intermediate NEB images
        moving_atom_idx: Index of the moving oxygen atom
        output_dir: Output directory for visualizations
        file_manager: FileManager instance for CONTCAR cleaning
    
    Returns:
        Dictionary of created file paths
    """
    visualizer = NEBPathVisualizer()
    return visualizer.create_all_visualizations(
        neb_dir, n_images, moving_atom_idx, output_dir, file_manager
    )


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("NEB Path Visualization Module")
    print("============================")
    print("This module creates visualizations of oxygen migration paths:")
    print("- 2D projections (xy, xz, yz planes)")
    print("- 3D visualizations with multiple viewing angles")
    print("- Sequential snapshots showing migration progression")
    print("\nOxygen atoms are colored:")
    print("- Initial position: Gold/Yellow (solid)")
    print("- Final position: Red (solid)")
    print("- Transition states: Orange (transparent with outline)")