<?php
require 'simple_html_dom.php';

$district=$_GET['district'];
$village=$_GET['village'];

$village=str_replace( "(","_3a", $village);
$village=str_replace(")","_4a", $village);
$url="http://www.onefivenine.com/india/villages/Solapur/".$district."/".$village;
$html = file_get_html($url);

$title = $html->find('div.boxinside2',4);
if($title==null)
{
	echo "0";
}
else
{
	$test=$title->plaintext;
	$test = str_replace('&nbsp;', '', $test);
	$test = str_replace('&deg;C', 'C,', $test);
	$test = str_replace('Weather', 'Weather,', $test);
	$test = str_replace('Humidity', ',Humidity', $test);
	$test = str_replace('Wind', ',Wind', $test);
	$test = str_replace('Station', ',Station', $test);
	$test = str_replace('observed', ',observed', $test);
	$test = str_replace('"', "", $test);
	$test = str_replace(PHP_EOL, '', $test);
	

	echo $test;

}


?>
