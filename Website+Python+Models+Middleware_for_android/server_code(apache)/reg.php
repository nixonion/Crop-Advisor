<?php


$date = $_GET['date'];
$crop= $_GET['crop'];

$list = array
(
",,,Commodity,,,,,,pricedate",
",,,$crop,,,,,,$date",
);



$file = fopen("test.csv","w");

foreach ($list as $line)
  {
  fputcsv($file,explode(',',$line));
  }

fclose($file); 
$text= shell_exec('python3 reg.py');
echo $text;

 
unlink("test.csv");
?>
