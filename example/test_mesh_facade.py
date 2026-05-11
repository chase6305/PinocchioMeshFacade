"""Example: PinocchioMeshFacade functionality demonstration and viser visualization."""

import time
from pathlib import Path

import numpy as np

from pinocchio_mesh_facade.pinocchio_mesh_facade import PinocchioMeshFacade


class ViserVisualizer:
    """Viser visualization manager."""

    def __init__(self):
        """Initialize the viser visualization manager."""
        try:
            import viser
            self.server = viser.ViserServer()
            self.scene = self.server.scene
            print("Viser server running at http://localhost:8081 (Ctrl+C to quit)")
            # Add /world frame at initialization
            self.enabled = True
            self._add_world_frame()
        except ImportError:
            self.server = None
            self.scene = None
            self.enabled = False

    def _add_world_frame(self):
        if self.enabled and self.scene:
            # Check if /world already exists to avoid duplication
            frame_names = []
            if hasattr(self.scene, "get_frame_names"):
                try:
                    frame_names = list(self.scene.get_frame_names())
                except Exception:
                    frame_names = []
            elif hasattr(self.scene, "frames"):
                try:
                    frame_names = list(self.scene.frames.keys())
                except Exception:
                    frame_names = []
            if "/world" not in frame_names:
                self.scene.add_frame(
                    "/world",
                    wxyz=(1.0, 0.0, 0.0, 0.0),
                    position=(0.0, 0.0, 0.0),
                    axes_length=0.5,
                    axes_radius=0.002,
                )

    def wait_step(self, msg):
        """Wait for the user to press Enter to continue."""
        if self.enabled:
            input(f"\n[viser] {msg} Press Enter to continue...")

    def clear_scene(self):
        """Clear the current scene."""
        if not self.enabled:
            return
        scene = self.scene
        # 1. Prefer new API: get_mesh_names/remove_mesh, get_frame_names/remove_frame
        try:
            if hasattr(scene, "get_mesh_names") and hasattr(scene, "remove_mesh"):
                for name in list(scene.get_mesh_names()):
                    scene.remove_mesh(name)
            # Remove frames except /world
            if hasattr(scene, "get_frame_names") and hasattr(scene, "remove_frame"):
                for name in list(scene.get_frame_names()):
                    if name != "/world":
                        scene.remove_frame(name)
                self._add_world_frame()
                return
        except Exception:
            pass
        # 2. Compatible with old API: meshes/frames attributes
        try:
            if hasattr(scene, "meshes") and hasattr(scene, "remove_mesh"):
                for name in list(scene.meshes.keys()):
                    scene.remove_mesh(name)
            if hasattr(scene, "frames") and hasattr(scene, "remove_frame"):
                for name in list(scene.frames.keys()):
                    if name != "/world":
                        scene.remove_frame(name)
                self._add_world_frame()
                return
        except Exception:
            pass
        # 3. Try clear/reset methods
        if hasattr(scene, "clear"):
            try:
                scene.clear()
            except Exception:
                pass
        elif hasattr(scene, "reset"):
            try:
                scene.reset()
            except Exception:
                pass
        # Always ensure /world exists
        self._add_world_frame()

    def add_mesh_simple(self, name, vertices, faces, color, wireframe):
        """Add a simple mesh to the scene."""
        if self.enabled:
            self.scene.add_mesh_simple(
                name=name,
                vertices=vertices,
                faces=faces,
                color=color,
                wireframe=wireframe,
            )

    def finish(self):
        """Finish the visualization steps."""
        if self.enabled:
            print(
                "All visualization steps are complete. Close the window or press Ctrl+C to exit."
            )


if __name__ == "__main__":
    visualizer = ViserVisualizer()
    # visualizer.enabled = False  # 如需跳过可视化可取消注释

    # 配置参数化
    robot_cfgs = [
        {
            "name": "hand",
            "urdf": lambda: str(
                Path(__file__).parent / "BrainCo_Revo1_R/left_hand/left_hand.urdf"
            ),
            "output_dir": lambda: Path(__file__).parent.parent / "output" / "hand",
            "joint_angles": {
                "T_CMC_YAW": np.deg2rad(30.0),
                "T_MCP": np.deg2rad(30.0),
                "IF_MCP_PITCH": np.deg2rad(50.0),
                "MF_MCP_PITCH": np.deg2rad(50.0),
                "RF_MCP_PITCH": np.deg2rad(50.0),
                "LF_MCP_PITCH": np.deg2rad(50.0),
            },
            "main_link": "base_link",
            "other_link": "t_mcp_link",
            "merge_ref_link": "base_link",
            "joint_names": ["IF_MCP_PITCH", "MF_MCP_PITCH"],
            "root_link": "base_link",
        },
    ]

    def run_extractor_demo(cfg, visualizer):
        """Run a demonstration of PinocchioMeshFacade mesh extraction and visualization.

        :param cfg: Configuration dictionary for the demo
        :param visualizer: Visualizer instance
        """
        urdf_path = cfg["urdf"]()
        output_dir = cfg["output_dir"]()
        package_dirs = [Path(urdf_path).parent]

        extractor = PinocchioMeshFacade(urdf_path, package_dirs, verbose=True)
        extractor.update_joint_angle(cfg["joint_angles"])

        # 1. Get mesh paths for a specific link
        paths = extractor.get_mesh_paths_by_link(
            cfg["main_link"], include_visual=False, include_collision=True
        )
        print(f"(from link) {cfg['main_link']} collision paths num: {len(paths)}, paths: {paths}")
        paths = extractor.get_mesh_paths_by_joints(
            cfg["joint_names"], include_visual=False, include_collision=True
        )
        print(
            f"(from joints) {cfg['joint_names']} collision paths num: {len(paths)}, paths: {paths}"
        )
        if not paths:
            print(
                f"[Warning] No mesh paths found for link: {cfg['main_link']} "
                f"(include_visual=False, include_collision=True)"
            )

        if visualizer.enabled:
            visualizer.clear_scene()
            meshes = extractor.get_meshes_by_link(
                cfg["main_link"], include_visual=False, include_collision=True
            )
            for i, m in enumerate(meshes):
                vertices = np.asarray(m.mesh.vertices, dtype=np.float32)
                faces = np.asarray(m.mesh.triangles, dtype=np.int32)
                visualizer.add_mesh_simple(
                    name=f"step1_mesh_{i}",
                    vertices=vertices,
                    faces=faces,
                    color=(0.7, 0.7, 0.2),
                    wireframe=False,
                )
            visualizer.wait_step(
                f"Step 1: Mesh paths loaded and meshes visualized for '{cfg['main_link']}'."
            )

        # 2. Load mesh objects for another link
        meshes = extractor.get_meshes_by_link(
            cfg["other_link"], include_visual=False, include_collision=True
        )
        for m in meshes:
            print(f"loaded mesh: {m.link_name}, vertices={len(m.mesh.vertices)}")
        if visualizer.enabled:
            visualizer.clear_scene()
            for i, m in enumerate(meshes):
                vertices = np.asarray(m.mesh.vertices, dtype=np.float32)
                faces = np.asarray(m.mesh.triangles, dtype=np.int32)
                visualizer.add_mesh_simple(
                    name=f"mesh_{i}",
                    vertices=vertices,
                    faces=faces,
                    color=(0.2, 0.6, 1.0),
                    wireframe=False,
                )
            visualizer.wait_step(f"Step 2: Visualized meshes for '{cfg['other_link']}'.")

        # 3. Benchmark mesh loading (cold and warm cache)
        for cache_label in ["cold", "warm"]:
            start_time = time.time()
            meshes = extractor.get_meshes_by_link(
                cfg["main_link"], include_visual=False, include_collision=True
            )
            for m in meshes:
                print(
                    f"loaded {cfg['main_link']} mesh: {m.link_name}, "
                    f"vertices={len(m.mesh.vertices)}"
                )
            end_time = time.time()
            print(f"Loading meshes took {end_time - start_time:.5f} seconds ({cache_label} cache)")
            if visualizer.enabled:
                visualizer.clear_scene()
                color = (0.8, 0.4, 0.2) if cache_label == "cold" else (0.2, 0.8, 0.4)
                for i, m in enumerate(meshes):
                    vertices = np.asarray(m.mesh.vertices, dtype=np.float32)
                    faces = np.asarray(m.mesh.triangles, dtype=np.int32)
                    visualizer.add_mesh_simple(
                        name=f"{cache_label}_mesh_{i}",
                        vertices=vertices,
                        faces=faces,
                        color=color,
                        wireframe=False,
                    )
                visualizer.wait_step(
                    f"Step {3 if cache_label == 'cold' else 4}: Visualized {cache_label} "
                    f"cache meshes for '{cfg['main_link']}'."
                )

        # 5. Merge all meshes for a link
        start_time = time.time()
        meshes = extractor.get_meshes_by_link(
            cfg["main_link"], include_visual=False, include_collision=True, merge=True
        )
        for m in meshes:
            print(
                f"loaded {cfg['main_link']} merged mesh: {m.link_name}, "
                f"vertices={len(m.mesh.vertices)}"
            )
        end_time = time.time()
        print(f"Merging meshes took {end_time - start_time:.5f} seconds")
        if visualizer.enabled:
            visualizer.clear_scene()
            for i, m in enumerate(meshes):
                vertices = np.asarray(m.mesh.vertices, dtype=np.float32)
                faces = np.asarray(m.mesh.triangles, dtype=np.int32)
                visualizer.add_mesh_simple(
                    name=f"merged_mesh_{i}",
                    vertices=vertices,
                    faces=faces,
                    color=(0.6, 0.2, 0.8),
                    wireframe=False,
                )
            visualizer.wait_step(f"Step 5: Visualized merged mesh for '{cfg['main_link']}'.")

        # 6. Export merged mesh to file
        merged_path = extractor.export_merged_mesh_by_link(
            cfg["main_link"],
            output_dir / f"{cfg['main_link']}_merged.obj",
            include_visual=False,
            include_collision=True,
            reference_link=cfg["merge_ref_link"],
        )
        print(f"Exported merged mesh to {merged_path}")
        if visualizer.enabled:
            visualizer.clear_scene()
            meshes = extractor.get_meshes_by_link(
                cfg["main_link"], include_visual=False, include_collision=True, merge=True
            )
            for i, m in enumerate(meshes):
                vertices = np.asarray(m.mesh.vertices, dtype=np.float32)
                faces = np.asarray(m.mesh.triangles, dtype=np.int32)
                visualizer.add_mesh_simple(
                    name=f"merged_mesh_{i}",
                    vertices=vertices,
                    faces=faces,
                    color=(0.3, 0.3, 0.7),
                    wireframe=False,
                )
            visualizer.wait_step("Step 6: Exported merged mesh to file and visualized.")

        # 7. Export validation scene to file
        val_path = extractor.export_validation_scene_by_link(
            cfg["main_link"],
            output_dir / "validation.obj",
            include_visual=False,
            include_collision=True,
        )
        print(f"Exported validation scene to {val_path}")
        if visualizer.enabled:
            visualizer.clear_scene()
            meshes = extractor.get_meshes_by_link(
                cfg["main_link"], include_visual=False, include_collision=True
            )
            for i, m in enumerate(meshes):
                vertices = np.asarray(m.mesh.vertices, dtype=np.float32)
                faces = np.asarray(m.mesh.triangles, dtype=np.int32)
                visualizer.add_mesh_simple(
                    name=f"val_mesh_{i}",
                    vertices=vertices,
                    faces=faces,
                    color=(0.2, 0.7, 0.7),
                    wireframe=False,
                )
            visualizer.wait_step("Step 7: Exported validation scene to file and visualized.")

        # 8. Get current robot configuration
        current_q = extractor.get_current_configuration()
        print(f"Current configuration shape: {current_q.shape}")
        if visualizer.enabled:
            visualizer.clear_scene()
            meshes = extractor.get_meshes_by_link(
                cfg["main_link"], include_visual=False, include_collision=True
            )
            for i, m in enumerate(meshes):
                vertices = np.asarray(m.mesh.vertices, dtype=np.float32)
                faces = np.asarray(m.mesh.triangles, dtype=np.int32)
                visualizer.add_mesh_simple(
                    name=f"curq_mesh_{i}",
                    vertices=vertices,
                    faces=faces,
                    color=(0.7, 0.5, 0.2),
                    wireframe=False,
                )
            visualizer.wait_step("Step 8: Got current robot configuration and visualized.")

        # 9. Get mesh paths by joint names
        joint_names = cfg["joint_names"]
        mesh_paths_by_joint = extractor.get_mesh_paths_by_joints(joint_names)
        print(f"Mesh paths by joints: {mesh_paths_by_joint}")
        if visualizer.enabled:
            visualizer.clear_scene()
            mesh_resources_by_joint = extractor.get_meshes_by_joints(joint_names)
            for i, resources in enumerate(mesh_resources_by_joint):
                for j, m in enumerate(resources):
                    vertices = np.asarray(m.mesh.vertices, dtype=np.float32)
                    faces = np.asarray(m.mesh.triangles, dtype=np.int32)
                    visualizer.add_mesh_simple(
                        name=f"joint_{i}_mesh_{j}",
                        vertices=vertices,
                        faces=faces,
                        color=(0.5, 0.7, 0.3),
                        wireframe=False,
                    )
            visualizer.wait_step("Step 9: Got mesh paths by joint names and visualized.")

        # 10. Get mesh resources by joint names
        mesh_resources_by_joint = extractor.get_meshes_by_joints(joint_names)
        for i, resources in enumerate(mesh_resources_by_joint):
            print(f"Joint {joint_names[i]} has {len(resources)} mesh resources")
        if visualizer.enabled:
            visualizer.clear_scene()
            for i, resources in enumerate(mesh_resources_by_joint):
                for j, m in enumerate(resources):
                    vertices = np.asarray(m.mesh.vertices, dtype=np.float32)
                    faces = np.asarray(m.mesh.triangles, dtype=np.int32)
                    visualizer.add_mesh_simple(
                        name=f"joint_{i}_mesh_{j}",
                        vertices=vertices,
                        faces=faces,
                        color=(0.9, 0.3, 0.3),
                        wireframe=False,
                    )
            visualizer.wait_step("Step 10: Visualized mesh resources by joint names.")

        # 11. Merge all meshes for root link (visual)
        start_time = time.time()
        meshes = extractor.get_meshes_by_link(
            cfg["root_link"], include_visual=True, include_collision=False, merge=True
        )
        for m in meshes:
            print(
                f"loaded {cfg['root_link']} merged mesh: {m.link_name}, "
                f"vertices={len(m.mesh.vertices)}"
            )
        end_time = time.time()
        print(f"Merging meshes took {end_time - start_time:.5f} seconds")
        if visualizer.enabled:
            visualizer.clear_scene()
            for i, m in enumerate(meshes):
                vertices = np.asarray(m.mesh.vertices, dtype=np.float32)
                faces = np.asarray(m.mesh.triangles, dtype=np.int32)
                visualizer.add_mesh_simple(
                    name=f"merged_mesh_{i}",
                    vertices=vertices,
                    faces=faces,
                    color=(0.5, 0.5, 0.5),
                    wireframe=False,
                )
            visualizer.wait_step(f"Step 11: Visualized merged mesh for root '{cfg['root_link']}'.")

        # 12. Export merged mesh to stl file
        merged_path = extractor.export_merged_mesh_by_link(
            cfg["root_link"],
            output_dir / f"{cfg['name']}_merged.stl",
            include_visual=True,
            include_collision=False,
            reference_link=cfg["root_link"],
            file_type="stl",
        )
        print(f"Exported merged mesh to {merged_path}")
        if visualizer.enabled:
            visualizer.clear_scene()
            meshes = extractor.get_meshes_by_link(
                cfg["root_link"], include_visual=True, include_collision=False, merge=True
            )
            for i, m in enumerate(meshes):
                vertices = np.asarray(m.mesh.vertices, dtype=np.float32)
                faces = np.asarray(m.mesh.triangles, dtype=np.int32)
                visualizer.add_mesh_simple(
                    name=f"merged_mesh_{i}",
                    vertices=vertices,
                    faces=faces,
                    color=(0.5, 0.5, 0.5),
                    wireframe=False,
                )
            visualizer.wait_step("Step 12: Exported merged mesh to stl file and visualized.")

        # 13. Export merged mesh to glb file
        merged_path = extractor.export_merged_mesh_by_link(
            cfg["root_link"],
            output_dir / f"{cfg['name']}_merged.glb",
            include_visual=True,
            include_collision=False,
            reference_link=cfg["root_link"],
            file_type="glb",
        )
        print(f"Exported merged mesh to {merged_path}")
        if visualizer.enabled:
            visualizer.clear_scene()
            visualizer.wait_step(
                "Step 13: Exported merged mesh to glb file and visualized. (no scene update)"
            )

        # End prompt
        if visualizer.enabled:
            visualizer.finish()
        else:
            print(
                "[viser] viser is not installed. "
                "Install with 'pip install viser' to enable web visualization."
            )

    for cfg in robot_cfgs:
        print(f"\n===== Running demo for {cfg['name']} =====")
        run_extractor_demo(cfg, visualizer)
