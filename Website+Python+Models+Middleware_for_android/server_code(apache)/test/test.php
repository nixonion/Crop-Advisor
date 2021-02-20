<?php
error_reporting(E_ALL);
$servername = "crop.cmotgzvfocad.us-east-2.rds.amazonaws.com";
$username = "root";
$password = "qwertyuiop";
$dbname = "crop";

// Create connection
$conn1 = new mysqli($servername, $username, $password, $dbname);
// Check connection
if ($conn1->connect_error) {
    die("Connection failed: " . $conn1->connect_error);
}

$area = $_GET['area'];
$ph= $_GET['ph'];
$temp=$_GET['temp'];
$taluka=$_GET['taluka'];
$village=$_GET['village'];
$water = $_GET['water'];
$soil = $_GET['soil'];
$lat = $_GET['lat'];
$lon = $_GET['lon'];

$sql="INSERT INTO crop VALUES ($area,$water,'$soil',$lon,$lat,$temp,$ph,'$taluka','$village');";

if ($conn1->query($sql) === TRUE ) {
    //echo "Inserted into Database :P";
} else {
    //echo "Error creating table: " . $conn1->error;
}


$conn1->close();


$list = array
(
"Crop ,Temperature,Water (Rainfall) mm,pH (Range),Soil Type,Area (ha),Yield (t),Growing Season",
",$temp,$water,$ph,$soil,,,",
);



$file = fopen("Input.csv","w");

foreach ($list as $line)
  {
  fputcsv($file,explode(',',$line));
  }

fclose($file);
echo "Crops You Can Grow:
"; 
echo shell_exec('python3 test.py');
//echo "DOne";

?>



<?php
 
unlink("Input.csv");
?>
