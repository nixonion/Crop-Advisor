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
$val1 = $_GET['name'];
$val2 = $_GET['name1'];
$val3 = $_GET['name2'];

$sql="INSERT INTO nabeel VALUES ('$val1','$val2','$val3');";

if ($conn1->query($sql) === TRUE ) {
    echo "<br>Table created successfully\n";
} else {
    echo "Error creating table: " . $conn1->error;
}
/*if ($conn1->multi_query($sql1) === TRUE ) {
    echo "<br>Table created successfully";
} else {

    echo "Error creating table: " . $conn1->error;
}*/
$response="Yayayayayayay";
echo json_encode($response);

$conn1->close();

?>
