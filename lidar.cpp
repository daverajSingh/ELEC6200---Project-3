/**********************************************************************
 Copyright (c) 2020-2023, Unitree Robotics.Co.Ltd. All rights reserved.
***********************************************************************/

#include "unitree_lidar_sdk.h"
#include "unitree_lidar_sdk_pcl.h"
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
using namespace unitree_lidar_sdk;

int main(){

  // Initialize Lidar Object
  UnitreeLidarReader* lreader = createUnitreeLidarReader();
  int cloud_scan_num = 18;
  std::string port_name = "/dev/ttyUSB0";

  if ( lreader->initialize(cloud_scan_num, port_name) ){
    printf("Unilidar initialization failed! Exit here!\n");
    exit(-1);
  }else{
    printf("Unilidar initialization succeed!\n");
  }

  // Set Lidar Working Mode
  printf("Set Lidar working mode to: STANDBY ... \n");
  lreader->setLidarWorkingMode(STANDBY);
  sleep(1);

  printf("Set Lidar working mode to: NORMAL ... \n");
  lreader->setLidarWorkingMode(NORMAL);
  sleep(1);
  
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);

  printf("\n");

  // Print Lidar Version
  while(true){
    if (lreader->runParse() == VERSION){
      printf("lidar firmware version = %s\n", lreader->getVersionOfFirmware().c_str() );
      break;
    }
    usleep(500);
  }
  printf("lidar sdk version = %s\n\n", lreader->getVersionOfSDK().c_str());
  sleep(2);

  // Check lidar dirty percentange
  int count_percentage = 0;
  while(true){
    if( lreader->runParse() == AUXILIARY){
      printf("Dirty Percentage = %f %%\n", lreader->getDirtyPercentage());
      if (++count_percentage > 2){
        break;
      }
      if (lreader->getDirtyPercentage() > 10){
        printf("The protection cover is too dirty! Please clean it right now! Exit here ...\n");
        exit(0);
      }
    }
    usleep(500);
  }
  printf("\n");
  sleep(2);

  // Set LED
  printf("Turn on all the LED lights ...\n");
  uint8_t led_table[45];
  for (int i=0; i < 45; i++){
    led_table[i] = 0xFF;
  }
  lreader->setLEDDisplayMode(led_table);
  sleep(2);

  printf("Turn off all the LED lights ...\n");
  for (int i=0; i < 45; i++){
    led_table[i] = 0x00;
  }
  lreader->setLEDDisplayMode(led_table);
  sleep(2);

  printf("Set LED mode to: FORWARD_SLOW ...\n");
  lreader->setLEDDisplayMode(FORWARD_SLOW);
  sleep(2);

  printf("Set LED mode to: REVERSE_SLOW ...\n");
  lreader->setLEDDisplayMode(REVERSE_SLOW);
  sleep(2);

  printf("Set LED mode to: SIXSTAGE_BREATHING ...\n");
  lreader->setLEDDisplayMode(SIXSTAGE_BREATHING);

  printf("\n");
  sleep(2);
  
  // Declare the PCL point cloud to accumulate points
    pcl::PointCloud<pcl::PointXYZI>::Ptr accumulated_cloud(new pcl::PointCloud<pcl::PointXYZI>);

  // Parse PointCloud and IMU data
  MessageType result;
  std::string version;
  while (true)
  {
    result = lreader->runParse(); // You need to call this function at least 1500Hz

    switch (result)
    {
    case NONE:
      break;
    
    case IMU:
      printf("An IMU msg is parsed!\n");
      printf("\tstamp = %f, id = %d\n", lreader->getIMU().stamp, lreader->getIMU().id);
      printf("\tquaternion (x, y, z, w) = [%.4f, %.4f, %.4f, %.4f]\n", 
              lreader->getIMU().quaternion[0], lreader->getIMU().quaternion[1], 
              lreader->getIMU().quaternion[2], lreader->getIMU().quaternion[3]);
      printf("\ttimedelay (us) = %d\n\n", lreader->getTimeDelay()); 
      break;
    
    case POINTCLOUD:
      printf("A Cloud msg is parsed! \n");
      printf("\tstamp = %f, id = %d\n", lreader->getCloud().stamp, lreader->getCloud().id);
      printf("\tcloud size  = %ld, ringNum = %d\n", lreader->getCloud().points.size(), lreader->getCloud().ringNum);
      printf("\tfirst 10 points (x,y,z,intensity,time,ring) = \n");
      for (int i = 0; i< 10; i++){ // print the first 10 points
        printf("\t  (%f, %f, %f, %f, %f, %d)\n", 
            lreader->getCloud().points[i].x, 
            lreader->getCloud().points[i].y, 
            lreader->getCloud().points[i].z, 
            lreader->getCloud().points[i].intensity, 
            lreader->getCloud().points[i].time, 
            lreader->getCloud().points[i].ring);
      }
      printf("\t  ...\n");
      printf("\ttimedelay (us) = %d\n\n", lreader->getTimeDelay());
      int num_points = 0;
      cloud->clear();
      for (const auto& point : lreader->getCloud().points) {
        pcl::PointXYZI pcl_point;
        pcl_point.x = point.x;
        pcl_point.y = point.y;
        pcl_point.z = point.z;
        pcl_point.intensity = point.intensity;

        cloud->points.push_back(pcl_point);
        accumulated_cloud->points.push_back(pcl_point);
        num_points++;
    }
    // Print the current accumulated number of points
    printf("Accumulated points: %lu\n", accumulated_cloud->points.size());

    cloud->width = cloud->points.size();
    cloud->height = 1;  // Unorganized point cloud
    cloud->is_dense = false;
    // Print some debug information
      printf("Expected points: %d, Added points: %lu\n", num_points, cloud->points.size());
    // Save to PCD file
    pcl::io::savePCDFileASCII("lidar_pointcloud.pcd", *cloud);
    printf("Saved %lu data points to lidar_pointcloud.pcd\n", cloud->points.size());
      break;

    default:
      break;
    }

    usleep(500);
  }
	// Set the final metadata for the accumulated cloud
    accumulated_cloud->width = accumulated_cloud->points.size();
    accumulated_cloud->height = 1;  // Unorganized point cloud
    accumulated_cloud->is_dense = false; // Assume it contains NaN values

    // Save the accumulated point cloud to a PCD file
    pcl::io::savePCDFileASCII("accumulated_lidar_pointcloud.pcd", *accumulated_cloud);
    printf("Saved %lu data points to accumulated_lidar_pointcloud.pcd\n", accumulated_cloud->points.size());
  return 0;
}