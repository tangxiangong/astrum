use std::collections::HashMap;

use nalgebra::Scalar;

/// Orientation style of mesh
pub enum OrientationStyle {
    /// Oriented mesh
    Oriented,
    /// Non-oriented mesh
    NonOriented,
}

/// Regularity style of mesh
pub enum RegularityStyle {
    /// Regular mesh
    Regular,
    /// Irregular mesh
    Irregular,
}

/// Base trait of mesh
pub trait Grid {
    /// Parameter space dimension
    const CELL_DIM: usize;

    /// Physical space dimension
    const POINT_DIM: usize;

    /// Node coordinate type
    type Coordinate: Scalar;

    /// Index type
    type Index;

    /// Reference element type
    type ReferenceElement;

    /// Point type using coordinate and dimension
    type Point;

    /// Get all node coordinates of the mesh
    fn node_coordinates(&self) -> &[Self::Point];

    /// Get cell-node connectivity of the mesh
    fn cell_node_ids(&self) -> &[Vec<Self::Index>];

    /// Get reference elements of the mesh
    fn reference_elements(&self) -> &[Self::ReferenceElement];

    /// Get cell types of the mesh
    fn cell_types(&self) -> &[u8];

    /// Get directionality style of the mesh
    fn orientation_style(&self) -> OrientationStyle {
        OrientationStyle::NonOriented
    }

    /// Get regularity style of the mesh
    fn regularity_style(&self) -> RegularityStyle {
        RegularityStyle::Regular
    }

    /// Get facet normal vector, only applicable to embedded space (POINT_DIM = CELL_DIM + 1)
    fn facet_normal(&self) -> Option<&[Self::Point]> {
        None
    }

    /// Get number of cells
    fn num_cells(&self) -> usize {
        self.cell_node_ids().len()
    }

    /// Get number of nodes
    fn num_nodes(&self) -> usize {
        self.node_coordinates().len()
    }
}

/// Grid topology trait
pub trait GridTopology {
    /// Parameter space dimension
    const CELL_DIM: usize;

    /// Physical space dimension
    const POINT_DIM: usize;

    /// Index type
    type Index;

    /// Polytope type
    type Polytope;

    /// Point type
    type Point;

    /// Get cell types
    fn cell_types(&self) -> &[u8];

    /// Get vertex coordinates
    fn vertex_coordinates(&self) -> &[Self::Point];

    /// Get polytopes
    fn polytopes(&self) -> &[Self::Polytope];

    /// Get connectivity from from_dim to to_dim
    fn faces(&self, from_dim: usize, to_dim: usize) -> &[Vec<Self::Index>];

    /// Get number of cells
    fn num_cells(&self) -> usize;

    /// Get number of faces of a specific dimension
    fn num_faces(&self, dim: usize) -> usize;

    /// Get total number of faces of all dimensions
    fn total_num_faces(&self) -> usize;

    /// Get directionality style
    fn orientation_style(&self) -> OrientationStyle {
        OrientationStyle::NonOriented
    }

    /// Get regularity style
    fn regularity_style(&self) -> RegularityStyle {
        RegularityStyle::Regular
    }

    /// Check if the mesh is oriented
    fn is_oriented(&self) -> bool {
        matches!(self.orientation_style(), OrientationStyle::Oriented)
    }

    /// Check if the mesh is regular
    fn is_regular(&self) -> bool {
        matches!(self.regularity_style(), RegularityStyle::Regular)
    }
}

#[allow(dead_code)]
/// Face labeling system
pub struct FaceLabeling {
    /// Number of faces of each dimension
    num_faces: Vec<usize>,

    /// Entity tags
    tags: Vec<u32>,

    /// Mapping from tag name to ID
    tag_name_to_id: HashMap<String, u32>,

    /// Mapping from tag ID to name
    tag_id_to_name: HashMap<u32, String>,

    /// Default tags of each dimension
    dim_to_default_tag: Vec<u32>,
}

impl FaceLabeling {
    /// Get the number of faces of a specific dimension
    pub fn num_faces(&self, dim: usize) -> usize {
        self.num_faces[dim]
    }

    /// Get the default tag of a specific dimension
    pub fn default_tag(&self, dim: usize) -> u32 {
        self.dim_to_default_tag[dim]
    }
}

/// Discrete model - mesh model for computation
pub trait DiscreteModel {
    /// Cell dimension
    const CELL_DIM: usize;

    /// Point dimension
    const POINT_DIM: usize;

    /// Grid type
    type Grid: Grid;

    /// Grid topology type
    type GridTopology: GridTopology;

    /// Get grid
    fn grid(&self) -> &Self::Grid;

    /// Get grid topology
    fn grid_topology(&self) -> &Self::GridTopology;

    /// Get face labeling
    fn face_labeling(&self) -> &FaceLabeling;
}
