#include <pcl/io/pcd_io.h> //Load PCD
#include <pcl/point_types.h> //POintcloud types
#include <pcl/filters/voxel_grid.h> //Downsampling (may not use)
#include <pcl/features/normal_3d.h> //Normal estimation
#include <pcl/search/kdtree.h> //Kdtree
#include <pcl/surface/gp3.h> //Greedy projection triangulation
#include <pcl/io/ply_io.h> //Saves mesh as ply

int main(int argc, char** argv) {
  //Load point cloud data from PCD file
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(argv[1], *cloud) == -1){
    PCL_ERROR("Couldn't read file %s \n",argv[1]);
    return(-1);
  }
  std::cout << "Loaded point cloud with " << cloud->size() << "points.\n";
  
  //Estimate normals
  pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
  pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
  normal_estimator.setSearchMethod(tree);
  normal_estimator.setInputCloud(cloud);
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  normal_estimator.setKSearch(100); //20 nearest neighbours
  normal_estimator.compute(*normals);
  std::cout << "Normals estimated.\n";
  
  //Combine XYZ and normal fields
  pcl::PointCloud<pcl::PointNormal>::Ptr cloud_with_normals(new pcl ::PointCloud<pcl::PointNormal>);
  pcl::concatenateFields(*cloud, *normals, *cloud_with_normals);
  
  //Create search tree for point cloud with normals
  pcl::search::KdTree<pcl::PointNormal>::Ptr tree2(new pcl::search::KdTree<pcl::PointNormal>);
  tree2->setInputCloud(cloud_with_normals);
  
  //Greedy Projection Triangulation
  pcl::GreedyProjectionTriangulation<pcl::PointNormal> gp3;
  pcl::PolygonMesh triangles;
  
  gp3.setSearchRadius(0.5); //Adjust search radius
  gp3.setMu(2.5);
  gp3.setMaximumNearestNeighbors(500);
  gp3.setMaximumSurfaceAngle(180);
  gp3.setMinimumAngle(10);
  gp3.setMaximumAngle(120);
  gp3.setNormalConsistency(false);
  
  //Generate surface mesh
  gp3.setInputCloud(cloud_with_normals);
  gp3.setSearchMethod(tree2);
  gp3.reconstruct(triangles);
  
  //Save as PLY
  pcl::io::savePLYFile("mesh.ply", triangles);
  std::cout << "Mesh saved to mesh.ply.\n";
  
  return 0;
}
