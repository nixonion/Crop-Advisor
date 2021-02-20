<?php 
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
$name = $_GET['crop'];
$area = $_GET['area'];
$ph= $_GET['ph'];
$temp=$_GET['temp'];
$taluka=$_GET['taluka'];
$village=$_GET['village'];
$water = $_GET['water'];
$soil = $_GET['soil'];
$lat = $_GET['lat'];
$lon = $_GET['lon'];

$sql="INSERT INTO crop VALUES ('$name',$area,$water,'$soil',$lat,$lon);";

if ($conn1->query($sql) === TRUE ) {
    echo "Inserted into Database :P";
} else {
    echo "Error creating table: " . $conn1->error;
}
/*if ($conn1->multi_query($sql1) === TRUE ) {
    echo "<br>Table created successfully";
} else {
    echo "Error creating table: " . $conn1->error;
}*/

$conn1->close();

?>
