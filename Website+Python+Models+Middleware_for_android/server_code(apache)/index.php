<!DOCTYPE html>
<html lang="en">

			


<?php
$servername = "crop.cmotgzvfocad.us-east-2.rds.amazonaws.com";
$username = "root";
$password = "qwertyuiop";
$dbname = "crop";

// Create connection
$conn1 = mysqli_connect($servername, $username, $password, $dbname);
// Check connection
if ($conn1->connect_error) {
    die("Connection failed: " . mysqli_connect_error());
}

if(isset($_POST['area']) && !empty($_POST["area"]))
{

$area = $_POST['area'];
$ph= $_POST['ph'];
$temp=$_POST['temp'];
$taluka=$_POST['taluka'];
$village=$_POST['village'];
$water = $_POST['water'];
$soil = $_POST['soil'];
$lat = $_POST['lat'];
$lon = $_POST['lon'];
//echo $lat;
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
$text= shell_exec('python3 test.py');
$text=str_replace("[", "", $text);
$text=str_replace("]", "", $text);
//echo $text;
//echo "Done";
echo "<script>alert('Crops you can grow are : ".json_encode($text)."')</script>";

 
//unlink("Input.csv");

}
if(isset($_POST['date']) && isset($_POST['cropfuture']) && !empty($_POST["date"] ) && !empty($_POST["cropfuture"]))
{
//echo "55";
$date = $_POST['date'];
$crop= $_POST['cropfuture'];

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
//echo $text;
echo "<script>alert('Crop price on the given date will be : ".json_encode($text)." per quintal')</script>";

 
//unlink("test.csv");



}
?>
<head>

    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <meta name="description" content="">
    <meta name="author" content="">

    <title>Crop Advisor</title>

    <!-- Bootstrap Core CSS -->
    <link href="vendor/bootstrap/css/bootstrap.min.css" rel="stylesheet">

    <!-- Custom Fonts -->
    <link href="vendor/font-awesome/css/font-awesome.min.css" rel="stylesheet" type="text/css">
    <link href="https://fonts.googleapis.com/css?family=Montserrat:400,700" rel="stylesheet" type="text/css">
    <link href='https://fonts.googleapis.com/css?family=Kaushan+Script' rel='stylesheet' type='text/css'>
    <link href='https://fonts.googleapis.com/css?family=Droid+Serif:400,700,400italic,700italic' rel='stylesheet' type='text/css'>
    <link href='https://fonts.googleapis.com/css?family=Roboto+Slab:400,100,300,700' rel='stylesheet' type='text/css'>

    <!-- Theme CSS -->
    <link href="css/agency.min.css" rel="stylesheet">

    <meta charset="utf-8" />
	<meta name="viewport" content="width=device-width, initial-scale=1.0">
	
	<link rel="shortcut icon" type="image/x-icon" href="docs/images/favicon.ico" />

    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.4.0/dist/leaflet.css" integrity="sha512-puBpdR0798OZvTTbP4A8Ix/l+A4dHDD0DGqYW6RQ+9jxkRFclaxxQb/SJAWZfWAkuyeQUytO7+7N4QKrDh+drA==" crossorigin=""/>
    <script src="https://unpkg.com/leaflet@1.4.0/dist/leaflet.js" integrity="sha512-QVftwZFqvtRNi0ZyCtsznlKSWOStnDORoefr1enyq5mVL4tmKB3S/EnC3rRJcxCPavG10IcrVGSmPh6Qw5lwrg==" crossorigin=""></script>
    <script src="https://ajax.googleapis.com/ajax/libs/jquery/3.3.1/jquery.min.js"></script>
    <script>

    	$("#Akluj").append("<option>1</option>");
    		



    	//$("#village").append($("<option>Temporary</option>"));//Akkalkot
    	var arraylist =["Akatnal","Akkalkot","Alage","Andewadi","Andewadi-Bk","Andewadi-Kh","Ankalage","Arali","Badole-Bk","Badole-Kh","Bagehalli","Banjgol","Barhanpur","Basalegaon","Basavgir","Bavkarwadi","Bhosgi","Bhurikavathe","Binjger","Boblad","Boregaon","Borgaon","Boroti-Bk","Boroti-Kh","Chapalgaon","Chapalgaonwadi","Chikkkahalli","Chincholi","Chungi","Dahitane","Dahitanewadi","Darshanal","Devikavathe","Dharsang","Dodyal","Dombar-Jawalge","Dudhani-","Galoragi","Gaudgaon-Bk","Gholasgaon","Ghungaregaon","Gogaon","Goudgaon-Kh","Guddewadi","Guravwadi","Haidre","Halchincholi","Halhalli","Handral","Hanjagi","Hannur","Hasapur","Hattikanbas","Hilli","Hingani","Ibrahimpur","Itage","Jainapur","Jakapur","Jeur","Jeurwadi","Kadabgaon","Kajikanbas","Kalappawadi","Kalegaon","Kalhipparage","Kalkarjal","Kanthehalli","Karajagi","Karjal","Kegaon-Bk","Kegaon-Kh","Khairat","Khanapur","Kini","Kiniwadi","Kirnalli","Kolekarwadi","Kolibet","Konhali","Korsegaon","Kudal","Kumthe","Kurnur","Mahalaxminagar","Maindargi","Mamdabad","Mangrul","Marathwadi","Matanhalli","Mhaisalage","Mhetre","Mirajgi","Motyal","Mugali","Mundhewadi","Naganhalli","Nagansur","Nagore","Nanhegaon","Navindgi","Nimgaon","Palapur","Parmanandnagar","Pitapur","Rampur","Ramtirth","Ruddewadi","Sadalapur","Safale","Salgar","Samarthnagar","Sangavi-Bk","Sangavi-Kh","Sangogi-","Sangogi-Aland","Satan-Dudhani","Sevalalnagar","Sevanagar","Shawal","Shegaon","Shirasi","Shirwal","Shirwalwadi","Sindkhed","Sinnur","Solase-Lamantanda","Sulerjawalaga","Sultanpur","Tadwal","Talewad","Tolnur","Torani","Udagi","Umarge","Vasantrao-Naik-Nagar-","Vijaynagar","Wagdari"];
    	var barshi=["Agalgaon","Alipur","Aljapur","Ambabaiwadi","Ambegaon","Arangaon","Babhulgaon","Balewadi","Barshi","Bavi","Belgaon","Bhalgaon","Bhandegaon","Bhansale","Bhatambare","Bhoinje","Bhoyare","Borgaon-(Zadi)","Borgaon-Kh.","Chare","Chikharde","Chinchkhopan","Chincholi","Chumb","Dadshinge","Dahitane","Devgaon","Dhamangaon-(A)","Dhamgaon-(Dumala)","Dhanore","Dhebarewadi","Dhorale","Dhotre","Gadegaon","Gatachiwadi","Gaudgaon","Ghanegaon","Ghari","Gholvewadi","Godasewadi","Gormale","Gulpoli","Halduge","Hattij","Hingani-(Pangaon)","Hingani-(Ratanjan)","Indapur","Irle","Irlewadi","Jahanpur","Jamgaon-(Agalgaon)","Jamgaon-(P)","Jawalgaon","Jotibachiwadi","Kalambawadi-(A)","Kalambwadi-(P)","Kalegaon","Kandalgaon","Kapasi","Kari","Kasari","Kasarwadi","Kategaon","Kavhe","Khadkalgaon","Khadkoni","Khamgaon","Khandavi","Koregaon","Korfale","Kuslamb","Ladole","Lakshyachiwadi","Mahagaon","Malegaon","Malegaon","Malwandi","Mamdapur","Mandegaon","Manegaon","Mirzanpur","Mouje-Tadwale","Mungashi-(R)","Mungashi-(Va)","Nagobachiwadi","Nandani","Nari-(Bhandewadi)","Nariwadi","Nimbalak","Pandhari","Pangaon","Pangri","Pathari","Phaphalwadi","Pimpalgaon-(Dhas)","Pimpalgaon-(Pangaon)","Pimpalgaon-(Pangri)","Pimpalwadi","Pimpari-(Pangaon)","Pimpri-(Ratanjan)","Puri","Raleras","Rastapur","Ratanjan","Raulgaon","Rui","Sakat","Sangamner","Sarjapur","Sarole","Sasure","Saundare","Sawargaon","Shelgaon-(Markad)","Shelgaon-(R)","Shelgaon-(Vhale)","Shendri","Shirale","Shripat-Pimpri","Surdi","Tadsoudne","Tambewadi","Tandulwadi","Tawadi","Turk-Pimpri","Ukadgaon","Umbarge","Undegaon","Upalai-(Thonge)","Upale-(Dumala)","Vairag","Wagachiwadi","Walwad","Wanewadi","Wangarwadi","Yawali","Yelamb","Zadi","Zaregaon"];

    	var karmala=["Aljapur","Alsunde","Anjandoh","Arjunnagar","Awati","Balewadi","Bhagatwadi","Bhalavni","Bhalewadi","Bhilarwadi","Bhose","Bitargaon-(Shrigonde)","Bitargaon-(Wangi)","Borgaon","Chikhalthan","Dahigaon","Delwadi","Deolali","Devichamal","Dhaykhindi","Dhokari","Dilmeshwar","Divegavan","Gaundare","Gharatwadi","Ghargaon","Ghoti","Gorewadi","Goyegaon","Gulmarwadi","Gulsadi","Hingani","Hisare","Hivare","Hiwarwadi","Hulgewadi","Jategaon","Jehurwadi","Jeur","Jinnti","Kamone","Kandar","Karanje","Karmala-(M-Cl)","Karmala-(Rural)","Katraj","Kavitgaon","Kawalwadi","Kedgaon","Kem","Ketur","Khadakewadi","Khadaki","Khambewadi","Khatgaon","Kolgaon","Kondhar-Chincholi","Kondhej","Korti","Kugaon","Kumbhargaon","Kumbhej","Kuskarwadi","Lavhe","Limbewadi","Malwadi","Mangi","Manjargaon","Mirghavan","Morwad","Nerle","Nilaj","Nimbhore","Nimgaon-(H)","Padali","Pande","Pangare","Parewadi","Pathurdi","Phisare","Pimpalwadi","Pomalwadi","Pondhvadi","Pophalaj","Potegaon","Pothare","Punwar","Rajuri","Ramwadi","Ravgaon","Ritewadi","Roshewadi","Sade","Salse","Sangvi","Sarapdoh","Satoli","Savadi","Shelgaon-(K)","Shelgaon-(Wangi)","Shetphal","Sogaon","Sounde","Takali-(Rashin)","Taratgaon","Umrad","Undargaon","Vadshivane","Vanjarwadi","Veet","Vihal","Wadachiwadi","Wadgaonwadgaon-(north)","Wangi","Warkatne","Warkute","Washibe","Zare"];

    	var madha=["Adhegaon","Ahergaon","Akole-Bk","Akole-Kh.","Akulgaon","Akumbhe","Alegaon-Bk","Alegaon-Kh.","Ambad","Anjangaon-Kh","Anjangaon-Umate","Aran","Badalewadi","Bairagwadi","Barloni","Bavi","Bembale","Bhend","Bhogewadi","Bhosare","Bhuinje","Bhutashte","Bitergaon","Budrukwadi","Chandaj","Chavhanwadi","Chavhanwadi","Chinchgaon","Chincholi","Chobhepimpari","Dahiwali","Darfal","Dhanore","Dhavalas","Footjawalgaon","Gar-Akole","Gavalewadi","Ghatane","Ghoti","Hatkarwadi","Hole-Kh.","Jadhavwadi","Jadhavwadi","Jakhale","Jamgaon","Kanhergaon","Kapsewadi","Kavhe","Kewad","Khairao","Khairewadi","Kumbhej","Kurdu","Lahu","Laul","Londhewadi","Loni","Madha","Mahadeowadi","Mahatpur","Malegaon","Manegaon","Mhaisgaon","Mitkalwadi","Modnimb","Mungashi","Nadi","Nagorli","Nimgaon","Padasali","Palwan","Panch-Phulwadi","Papnas","Parite","Paritewadi","Pimpalkhunte","Pimpalner","Randivewadi","Ranzani","Ridhore","Ropale-Kavhe","Ropale-Kh.","Rui","Sapatne-(Bhose)","Sapatne-Tembhurni","Shedshinge","Shevare","Shindewadi","Shingewadi","Shiral-Madha","Shiral-Tembhurni","Solankarwadi","Sultanpur","Surli","Tadavale","Takali-Tembhurni","Tambave","Tandulwadi","Tembhurni","Tulshi","Ujani-Madha","Ujani-Tembhurni","Undargaon","Upalai-Bk.","Upalai-Kh.","Upalawate","Venegaon","Vithalwadi","Wadachiwadi","Wadachiwadi","Wadachiwadi","Wadoli","Wadshinge","Wakav","Warawade","Wetalwadi"];

    	var malshiras=["Akluj","Anandnagar","Babhulgaon","Bacheri","Bagechiwadi","Bangarde","Bhamb","Bhamburdi","Bijvadi","Bondale","Borgaon","Chakore","Chandapuri","Chaundeshwarwadi","Dahigaon","Dasur","Dattanagar","Deshmukhwadi","Dhanore","Dharmpuri","Dombalwadi","Dombalwadi(khudus)","Ekshiv","Fadtari","Falwani","Fondshiras","Ganeshgaon","Garwad","Giravi","Girzani","Goradwadi","Gursale","Hanumanwadi","Islampur","Jadhavwadi","Jalbhavi","Jambud","Kacharewadi","Kadamwadi","Kalamboli","Kalamwadi","Kanher","Karunde","Khalawe","Khandali","Khudus","Kolegaon","Kondabavi","Kondarpatta","Kothale","Kurbavi","Kusmod","Lawang","Lonand","Londhe-Mohitewadi","Magarwadi","Mahalung","Malewadi","Malewadi","Malinagar","Malkhambi","Maloli","","Mandaki","Mandave","Markadwadi","Medad","Mire","Morochi","Motewadi","Motewadi","Natepute","Neware","Nimgaon","Nitavewadi","Palasmandal","Paniv","Pathanwasti","Piliv","Pimpari","Pirale","Pisewadi","Pratapnagar","Purandawade","Rede","Sadashivnagar","Salmukhwadi","Sangam","Sangramnagar","Savatgavhan","Shendechinch","Shindewadi","Shingorni","Shiwarvasti","Sulewadi","Tambave","Tambewadi","Tamsidwadi","Tandulwadi","Tarangfal","Tirwandi","Tondale","Ughadewadi","Umbare-(Velapur)","Umbare-Dahigaon","Vatpali","Velapur","Vijaywadi","Vithalwadi","Vizori","Wafegaon","Wagholi","Yashawantnagar","Yeliv","Zanjevasti","Zunjewadi"];

    	var mangalvedhe=["Akole","Andhalgaon","Arali","Asabewadi","Bathan","Bavachi","Bhalewadi","Bhalwani","Bhose","Borale","Bramhapuri","Chikhalgi","Degaon","Dharamgaon","Dhavalas","Diksal","Donaj","Dongargaon","Fatewadi","Ganeshwadi","Gharniki","Gonewadi","Gunjegaon","Hajapur","Hivargaon","Huljanti","Hunnar","Jalihal","Jangalgi","Jitti","Junoni","Kacharewadi","Kagasht","Karjal","Katral","Khadaki","Khave","Khomnal","Khupsangi","Lamantanda","Lavangi","Laxami-Dahiwadi","Lendave-Chichale","Lonar","Machanur","Mahamadabad-(h)","Mahamadabad-(s)","Malewadi","Mallewadi","Manewadi","Mangalvedha","Marapur","Maravade","Maroli","Metkarwadi","Mudhavi","Mundhewadi","Nandeshwar","Nandur","Nimboni","Padolkarwadi","Patkhal","Pout","Radde","Rahatewadi","Revewadi","Salagar-Bk","Salagar-Kh","Shelewadi","Shirashi","Shirnadgi","Shivangi","Siddapur","Siddhankeri","Soddi","Talsangi","Tamdardi","Tandor","Uchethan","Yedrav","Yelagi"];

    	var mohol=["Adhegaon","Angar","Ankoli","Arbali","Ardhanari","ArjunSond","Ashte","Ashti","Aundhi","Bairagwadi","Bhairowadi","Bhambewadi","Bhoire","Bitle","Bople","Chikhali","Chincholikati","Dadapur","Degaon","Deodi","Dhaingade-Wadi","Dhokbabulgaon","Diksal","Ekurke","Galandawadi","Ghatne","Ghodeshwar","Ghorpadi","Gotewadi","Haralwadi","Hingani(Nipani)","Hivare","Ichgaon","Jamgaon-Bk","Jamgaon-Kh","Kamti-Bk","Kamti-Kh","Katewadi","Khandali","Khandobachiwadi","Kharkatne","Khavani","Khuneshwar","Kolegaon","Kombadwadi","Konheri","Korwali","Kothale","Kuranwadi","Kuranwadi-(Ashti)","Kurul","Lamantanda","Lamboti","Malikpeth","Mangaoli","Maslechaudhari","Miri","Mohol","Morvanchi","Mundhewadi","Najikpimpari","Nalbandwadi","Nandgaon","Narkhed","Papari","Parmeshwar-pimpri","Paslewadi","Patkul","Pawarwadi","Peertakali","Penur","Pokharapur","Pophali","Ramhingani","Sarole","Saundane","Sawaleshwar","Sayyadwarwade","Shejbabhulgaon","Shetphal","Shingoli","Shirapur-(Mo)","Shirapur-(Solapur)","Siddewadi","Sohale","Takali-(Shikandar)","Tambole","Taratgaon","Telangwadi","Wadachiwadi","Waddegaon","Wadwal","Wafale","Wagholi","Wagholiwadi","Waluj","Warkute","Watwate","Wirawade-Bk","Wirwade-Kh","Yawali","Yellamwadi","Yenaki","Yeoti"];

    	var pandharpur=["Adhiv","Ajansond","Ajote","Ambe","Ambechincholi","Anawali","Avhe","Babhulgaon","Badalkot","Bardi","Bhalawani","Bhandi-Shegaon","Bhatumbare","Bhose","Bitargaon","Bohali","Chale","Chilaiwadi","Chincholi-Bhose","Chinchumbe","Degaon","Devade","Dhondewadi","Eklaspur","Fulchincholi","Gadegaon","Gardi","Gopalpur","Gurasale","Hole","Isbavi","Ishwar-Wathar","Jadhavwadi","Jainwadi","Jaloli","Kanhapuri","Karkamb","Karole","Kasegaon","Kauthali","Keskarwadi","Kharatwadi","Khardi","Kharsoli","Khed-Bhalawani","Khed-Bhose","Kondharki","Korty","Lonarwadi","Magarwadi","Mendhapur","Mundhewadi","Nali","Nandore","Narayan-Chincholi","Nematwadi","Nepatgaon","Ozewadi","Palshi","Pandharewadi","Patvardhan-Kuroli","Pehe","Pirachi-Kuroli","Pohargaon","Puluj","Pulujwadi","Ranzani","Ropale","Sangavi","Sarkoli","Shankargaon","Sharadrenagar","Shegaon-Dumala","Shelve","Shendgewadi","Shetphal","Shevate","Shirgaon","Shirthon","Siddhewadi","Sonke","Sugavabhose","Sugavakhurd","Supli","Suste","Takaligursala","Takli","Tanali","Tarapur","Taratgaon-(Bhose)","Taratgaon-Kasegaon","Tavashi","Tisangi","Tungat","Ujani","Umbare","Umbargaon","Upari","Venunagar","Vite","Wadi-Kuroli","Wakhari"];

    	var sangole=["Achakadani","Aglavewadi","Ajanale","Akola","Alegaon","Ankadhal","Bagalwadi","Balvadi","Bamani","Bandgarwadi","Bandgarwadi","Bhopasewadi","Buddhehal","Buralewadi","Burange-Wadi","Chikmahud","Chinake","Chincholi","Chopadi","Devale","Devkatewadi","Dhalewadi","Dhayati","Diksal","Dongargaon","Ekhatapur","Galavewadi","Gavadewadi","Gaygavhan","Gheradi","Godasewadi","Goudwadi","Gunappawadi","Habisewadi","Haldahivadi","Hangirage","Hanmantgaon","Hatid","Hatkar-Mangewadi","Itaki","Jadhavwadi","Javala","Jujarpur","Juni-Lotewadi","Junoni","Kadlas","Kalubaluwadi","Kamalapur","Karadwadi","Karandewadi","Karandewadi","Karandewadi","Katfal","Kedarwadi","Khavaspur","Khilarwadi","Kidabisari","Kola","Kombadwadi","Laxminagar","Ligadewadi","Lonavire","Lotewadi","Mahim","Mahud-Bk.","Manegaon","Manjari","Medashingi","Metakarwadi","Methvade","Misalwadi","Nalvadewadi","Narale","Naralewadi","Nazare","Nijampur","Pachegaon-Bk.","Pachegaon-Kh.","Pare","Rajapur","Rajuri","Sangewadi","Sangola","Saragarwadi","Satarkarvasti","Save","Shirbavi","Shivane","Sonalwadi","Sonand","Sonewadi","Tarangewadi","Tippehali","Udanwadi","Vasud","Vazare","Wadegaon","Waki-Gherdi","Waki-Shivane","Wani-Chinchol","Watambare","Yelmar-Mangewadi","Zapachiwadi"];

    	var south_solapur=["Hipparge","Achegaon","Aherwadi","Akole-Mandrup","Alegaon","Antroli","Auj-Aherwadi","Auj-Mandrup","Aurad","Balgi","Bandalgi","Bankalgi","Barur","Basav-Nagar","Bhandar-Kavade","Birnal","Bolkavathe","Boramani","Borul","Chandrahal","Chinchpur","Chinoholi","Darganhalli","Dhotri","Dindur","Doddi","Gangewadi","Gavadewadi","Ghodatanda","Gunjegaon","Gurdehalli","Hanamgaon","Hattarsang","Hatur","Hipale","Honmurgi","Hotgi","Hotgi-Station","Indiranagar","Ingalgi","Kanbas","Kandalgaon","Kandehalli","Karkal","Kasegaon","Khanapur","Kudal","Kumbhari","Kurghot","Kusur","Lavangi","Madre","Malkavathe","Mandrup","Mangoli","Mulegaon","Mulegaon-Tanda","Musti","Nandani","Nimbargi","Phatatewadi","Pinjarwadi","Rajur","Rampur","Sadepur","Sangdari","Sanjwad","Savatkhed","Shankar-Nagar","Shingadgaon","Shirpanhalli","Shirval","Sindkhed","Takali","Tandulwadi","Telgaon-Mandrup","Tillehal","Tirth","Togarali","Ule","Ulewadi","Vadakbal","Vadapur","Valsang","Vinchur","Wadgaon","Wadji","Wangi","Waralegaon","Yelegaon","Yetnal"];

    	var solapur_north=["Akolekati","Banegaon","Belati","Bhagaiwadi","Bhatewadi","Bhogaon","Darfal-(bb)","Darphal-(Gawadi)","Dongaon","Ekrukh","Gulwanchi","Haglur","Hipparge","Hiraj","Honsal","Inchgaon","Kalman","Karamba","Kawathe","Khed","Kondi","Kouthali","Mardi","Mohitewadi","Nandur","Nannaj","Narotewadi","Padsali","Pakani","Pathari","Raleras","Ranmasle","Sakharewadi","Samshapur","Sevalalnager","Shivani","Solapur-North","Taratgaon","Telgaon","Tirhe","Wadala","Wangi"];


    		$(document).ready(function(){

    		
    			$("#village").hide();
    			
    			//$("#Akkalkot").prop("disabled",true);
    		  $("#taluka").change(function(){
    		  	var selectedValue = $(this).val();
    		    //alert("The text has been changed."+selectedValue);
    		    //$("#village").append("<option>1"+arraylist[0]+"</option>");
    		    switch (selectedValue) {

    		    	case "Akkalkot":
    		    		for (var i = 0; i <= arraylist.length; i++)
    		    		{
    		    			$("#village").append("<option value='"+arraylist[i]+"''>"+arraylist[i]+"</option>");	
    		    		}
                        
                        break;
                    case "Barshi":
                    	for (var i = 0; i <= barshi.length; i++)
                    	{
                    		$("#village").append("<option value='"+barshi[i]+"''>"+barshi[i]+"</option>");	
                    	}
                        
                        break;
                    case "Karmala":
                    	for (var i = 0; i <= karmala.length; i++)
                    	{
                    		$("#village").append("<option value='"+karmala[i]+"''>"+karmala[i]+"</option>");	
                    	}
                        
                        break;
                    case "Madha":
                    	for (var i = 0; i <= madha.length; i++)
                    	{
                    		$("#village").append("<option value='"+madha[i]+"''>"+madha[i]+"</option>");	
                    	}
                        
                        break;
                    case "Malshiras":
                        for (var i = 0; i <= malshiras.length; i++)
    		    		{
    		    			$("#village").append("<option value='"+malshiras[i]+"''>"+malshiras[i]+"</option>");	
    		    		}
    		    		break;
                    case "Mangalvedhe":
                        for (var i = 0; i <= mangalvedhe.length; i++)
    		    		{
    		    			$("#village").append("<option value='"+mangalvedhe[i]+"''>"+mangalvedhe[i]+"</option>");	
    		    		}
    		    		break;
                    case "Mohol":
                        for (var i = 0; i <= mohol.length; i++)
    		    		{
    		    			$("#village").append("<option value='"+mohol[i]+"''>"+mohol[i]+"</option>");	
    		    		}
    		    		break;
                    case "Pandharpur":
                        for (var i = 0; i <= pandharpur.length; i++)
    		    		{
    		    			$("#village").append("<option value='"+pandharpur[i]+"''>"+pandharpur[i]+"</option>");	
    		    		}
    		    		break;
                    case "Sangole":
                        for (var i = 0; i <= sangole.length; i++)
    		    		{
    		    			$("#village").append("<option value='"+sangole[i]+"''>"+sangole[i]+"</option>");	
    		    		}
    		    		break;
                    case "Solapur-North":
                        for (var i = 0; i <= solapur_north.length; i++)
    		    		{
    		    			$("#village").append("<option value='"+solapur_north[i]+"''>"+solapur_north[i]+"</option>");	
    		    		}
    		    		break;
                    case "South-Solapur":
                        for (var i = 0; i <= south_solapur.length; i++)
    		    		{
    		    			$("#village").append("<option value='"+south_solapur[i]+"''>"+south_solapur[i]+"</option>");	
    		    		}
    		    		break;
    		      
    		    }
    		    /*for (var i = 0; i <= arraylist.length; i++)
    		    {
    		    		$("#village").append("<option>"+arraylist[i]+"</option>");	
    		    }*/
    		    
    		    $("#village").show();
    		    //$("#Akkalkot").prop("disabled",false);
    		  });
    		});


    	
    	/*$(function () {
    	        $("#taluka").change(function () {
    	            var selectedText = $(this).find("option:selected").text();
    	            var selectedValue = $(this).val();
    	            alert("Selected Text: " + selectedText + " Value: " +selectedValue);
    	            for (var i = 0; i <= arrayList.length; i++) {
    	                    //$('#village').append('<option value="' + arrayList[i] + '">' + arrayList[i] + '</option>');
    	                }
    	        });
    	    });*/
    </script>


    
</head>


<body id="page-top" class="index" onLoad="getLocation()">

    <!-- Navigation -->
    <nav id="mainNav" class="navbar navbar-default navbar-custom navbar-fixed-top">
        <div class="container">
            <!-- Brand and toggle get grouped for better mobile display -->
            <div class="navbar-header page-scroll">
                <button type="button" class="navbar-toggle" data-toggle="collapse" data-target="#bs-example-navbar-collapse-1">
                    <span class="sr-only">Toggle navigation</span> Menu <i class="fa fa-bars"></i>
                </button>
                <a class="navbar-brand page-scroll" href="#page-top">Crop Advisor</a>
            </div>

            <!-- Collect the nav links, forms, and other content for toggling -->
            <div class="collapse navbar-collapse" id="bs-example-navbar-collapse-1">
                <ul class="nav navbar-nav navbar-right">
                    <li class="hidden">
                        <a href="#page-top"></a>
                    </li>
                    <li>
                        <a class="page-scroll" href="#about">About</a>
                    </li>
                    <li>
                        <a class="page-scroll" href="#team">Team</a>
                    </li>
                    <li>
                        <a class="page-scroll" href="#contact">Crop</a>
                    </li>
                </ul>
            </div>
            <!-- /.navbar-collapse -->
        </div>
        <!-- /.container-fluid -->
    </nav>
   

    <!-- Header -->
    <header> 
        <div class="container">
            <div class="intro-text">
                <div class="intro-lead-in"><font color="black">Welcome To Our Crop Advisor!</font></div>
                <p> <font size="6" color="black"> This is a GIS based system which maps various parameters and provides a beneficial result that benefits and advises farmers and also buyers </font></p>
                <a href="#mapid" class="page-scroll btn btn-xl">Continue</a>
            </div>
        </div>
    </header>

    
    
    <fieldset>
    <!-- Form Section -->
    <section id="contact">
        <div class="container">
            <div class="row">
                <div class="col-lg-12 text-center">
                    <h2 class="section-heading">Crop</h2>
                    <h3 class="section-subheading text-muted">We will suggest the right crop to you!</h3>
                </div>
            </div>
            <div class="row">
                <div class="col-lg-12">
                    <form action="<?php $_SERVER['PHP_SELF']?>" method="POST" novalidate>
                        <div class="row">
                            <div class="col-md-6" style="width: 50%;margin-left: 25%;">

                            	<div class="form-group">
                                    <input type="number" class="form-control" placeholder="Land area (Acres)" id="area" name="area" required data-validation-required-message="Please enter your land area.">
                                    <p class="help-block text-danger"></p>
                                </div>
                                <div class="form-group">
                                    <input type="number" class="form-control" placeholder="Water (Amount of Rainfall in mm)" id="water" name="water" required data-validation-required-message="Please enter water content.">
                                    <p class="help-block text-danger"></p>
                                </div>
                                <div class="form-group">
                                    <input type="number" class="form-control" placeholder="Temperature (Celsius)" id="temp" name="temp" required data-validation-required-message="Please enter temperature.">
                                    <p class="help-block text-danger"></p>
                                </div>
                                <div class="form-group">
                                    <input type="number" class="form-control" placeholder="PH of your soil" id="ph" name="ph" required data-validation-required-message="Please enter PH.">
                                    <p class="help-block text-danger"></p>
                                </div>
                                <input type="hidden" id="lon" name="lon" value="">
                                <input type="hidden" id="lat" name="lat" value="">
                                <div class="form-group">
                                	<label class="container" style="display: block;
																	position: relative;
  																	padding-left: 35px;
  																	margin-bottom: 12px;
  																	cursor: pointer;
  																	font-size: 20px;
  																	color: white;"	>
                                	<p class="section-subheading text-muted" style="font-size: 22px;">Select soil type:</p>
                                	<input type="radio" name="soil" value="Medium-Deep" checked> Medium Deep<br>
                                	<input type="radio" name="soil" value="Deep-Black"> Deep Black<br>
                                  	<input type="radio" name="soil" value="Shallow"> Shallow<br>
                                	</label>
                                </div>
                                <div class="form-group">
                                	<label class="container" style="display: block;
																	position: relative;
  																	padding-left: 35px;
  																	text-align: center;
  																	margin-bottom: 12px;
  																	font-size: 20px;">
                                	<p class="section-subheading text-muted" style="font-size: 22px;">Select your location:</p>
	                                <select id="taluka" name="taluka">
	                                <option value="Akkalkot">Akkalkot</option>
	                                <option value="Barshi">Barshi</option>
	                                <option value="Karmala">Karmala</option>
	                                <option value="Madha">Madha</option>
	                                <option value="Malshiras">Malshiras</option>
	                                <option value="Mangalvedhe">Mangalvedhe</option>
	                                <option value="Mohol">Mohol</option>
	                                <option value="Pandharpur">Pandharpur</option>
	                                <option value="Sangole">Sangole</option>
	                                <option value="Solapur-North">Solapur-North</option>
	                                <option value="South-Solapur">South-Solapur</option>
                                	</select>
                                	<br><br>
                                	<select id="village" name="village"></select>
                                	
                            	</label>
                            	</div>
                            </div>
                            
                            <div class="clearfix"></div>
                            <div class="col-lg-12 text-center">
                                <div id="success"></div>
                                <button type="submit" class="btn btn-xl">Suggest</button>
                            </div>
                        </div>
                    </form><br><br><br><br><hr><br><br><br><br><br>
                    <form method="POST" action="<?php $_SERVER['PHP_SELF']?>" novalidate>
                        <div class="row">
                            <div class="col-md-6" style="width: 30%;margin-left: 35%;">

                                <div class="form-group">
                                    <input type="date" data-date-inline-picker="true" class="form-control" placeholder="" id="date" name="date" required data-validation-required-message="Please enter date.">
                                    <p class="help-block text-danger"></p>
                                </div>
                                <div class="form-group">
                                    <label class="container" style="display: block;
                                                                    position: relative;
                                                                    padding-left: 35px;
                                                                    text-align: center;
                                                                    margin-bottom: 12px;
                                                                    font-size: 20px;">
                                    <p class="section-subheading text-muted" style="font-size: 22px;">Select Crop:</p>
                                    <select id="cropfuture" name="cropfuture">
                                    <option value="Jowar">Jowar</option>
                                    <option value="Maize">Maize</option>
                                    <option value="Bajra">Bajra</option>
                                    <option value="Wheat">Wheat</option>
                                    </select>
                                    <br><br>
                                    
                                </label>
                                </div>
                                
                            
                            </div>
                            
                            <div class="clearfix"></div>
                            <div class="col-lg-12 text-center">
                                <div id="success"></div>
                                <button type="submit" class="btn btn-xl">Find Price</button>
                            </div>
                        </div>
                    </form>

                </div>
            </div>
        </div>
    </section>
    





<br><br><br><center> <p id="demo"></p></center>

</fieldset>






<div id="mapid" style="width: 96%;margin-left: 2%; height: 600px;"></div>
<script>

	var mymap = L.map('mapid').setView([17.902955, 75.871582], 9);

	L.tileLayer('https://api.tiles.mapbox.com/v4/{id}/{z}/{x}/{y}.png?access_token=pk.eyJ1IjoibWFwYm94IiwiYSI6ImNpejY4NXVycTA2emYycXBndHRqcmZ3N3gifQ.rJcFIG214AriISLbB6B5aw', {
		maxZoom: 18,
		attribution: 'Map data &copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a> contributors, ' +
			'<a href="https://creativecommons.org/licenses/by-sa/2.0/">CC-BY-SA</a>, ' +
			'Imagery  <a href="https://www.mapbox.com/">Mapbox</a>',
		id: 'mapbox.streets'
	}).addTo(mymap);

	

	

	L.marker([18.2334, 75.6941]).addTo(mymap).bindPopup("<b>Barshi</b><br />Area: 1,433.1 km2<br>Population:600,000<br>villages: 150<br> Avg. Rainfall: 594.8 mm<br><br><form action='villages.php' method='post'><input type='hidden'  value='Barshi' name='tehsil'><input type='hidden'  value='18.2334' name='lat'><input type='hidden'  value='75.6941' name='lon'><input type='submit' class='page-scroll btn btn-xl' style='border-radius: 3px;font-size: 10px; padding: 4px 6px ; '  value='Explore Barshi'></form>").openPopup();
	L.marker([17.5242, 76.2054]).addTo(mymap).bindPopup("<b>Akkalkot</b><br />Area: 1,390.3 km2<br>Population:314,570<br>villages: 138<br> Avg. Rainfall: 643.6 mm<br><br><form action='villages.php' method='post'><input type='hidden'  value='Akkalkot' name='tehsil'><input type='hidden'  value='17.5242' name='lat'><input type='hidden'  value='76.2054' name='lon'><input type='submit' class='page-scroll btn btn-xl' style='border-radius: 3px;font-size: 10px; padding: 4px 6px ; '  value='Explore Akkalkot'></form>").openPopup();
	L.marker([18.4045,75.1954]).addTo(mymap).bindPopup("<b>Karmala</b><br />Area: 1,609.7 km2<br>Population:254,489<br>villages: 122<br> Avg. Rainfall: 506 mm<br><br><form action='villages.php' method='post'><input type='hidden'  value='Karmala' name='tehsil'><input type='hidden'  value='18.4045' name='lat'><input type='hidden'  value='75.1954' name='lon'><input type='submit' class='page-scroll btn btn-xl' style='border-radius: 3px;font-size: 10px; padding: 4px 6px ; '  value='Explore Karmala'></form>").openPopup();
	L.marker([18.0340, 75.5161]).addTo(mymap).bindPopup("<b>Madha</b><br />Area: 1,544.9 km2<br>Population:292,611<br>villages: 117<br> Avg. Rainfall: 519 mm<br><br><form action='villages.php' method='post'><input type='hidden'  value='Madha' name='tehsil'><input type='hidden'  value='18.0340' name='lat'><input type='hidden'  value='75.5161' name='lon'><input type='submit' class='page-scroll btn btn-xl' style='border-radius: 3px;font-size: 10px; padding: 4px 6px ; '  value='Explore Madha'></form>").openPopup();
	L.marker([17.8633, 74.9055]).addTo(mymap).bindPopup("<b>Malshiras</b><br />Area:  km2<br>Population: 21,985<br>villages: 117 <br> Avg. Rainfall:  mm<br><br><form action='villages.php' method='post'><input type='hidden'  value='Malshiras' name='tehsil'><input type='hidden'  value='17.8633' name='lat'><input type='hidden'  value='74.9055' name='lon'><input type='submit' class='page-scroll btn btn-xl' style='border-radius: 3px;font-size: 10px; padding: 4px 6px ; '  value='Explore Malshiras'></form>").openPopup();
	L.marker([17.5110, 75.4520]).addTo(mymap).bindPopup("<b>Mangalvedhe</b><br />Area:  km2<br>Population: 21,694 <br>villages: 82<br> Avg. Rainfall:  mm<br><br><form action='villages.php' method='post'><input type='hidden'  value='Mangalvedhe' name='tehsil'><input type='hidden'  value='17.5110' name='lat'><input type='hidden'  value='75.4520' name='lon'><input type='submit' class='page-scroll btn btn-xl' style='border-radius: 3px;font-size: 10px; padding: 4px 6px ; '  value='Explore Mangalvedhe'></form>").openPopup();
	L.marker([17.8100, 75.6432]).addTo(mymap).bindPopup("<b>Mohol</b><br />Area:  km2<br>Population: 150,000<br>villages: 104<br> Avg. Rainfall:  mm<br><br><form action='villages.php' method='post'><input type='hidden'  value='Mohol' name='tehsil'><input type='hidden'  value='17.8100' name='lat'><input type='hidden'  value='75.6432' name='lon'><input type='submit' class='page-scroll btn btn-xl' style='border-radius: 3px;font-size: 10px; padding: 4px 6px ; '  value='Explore Mohol'></form>").openPopup();
	L.marker([17.6746, 75.3237]).addTo(mymap).bindPopup("<b>Pandharpur</b><br />Area: 25 km2<br>Population: 150,000<br>villages: 103 <br> Avg. Rainfall:  mm<br><br><form action='villages.php' method='post'><input type='hidden'  value='Pandharpur' name='tehsil'><input type='hidden'  value='17.6746' name='lat'><input type='hidden'  value='75.3237' name='lon'><input type='submit' class='page-scroll btn btn-xl' style='border-radius: 3px;font-size: 10px; padding: 4px 6px ; '  value='Explore Pandharpur'></form>").openPopup();
	L.marker([17.4341, 75.1954]).addTo(mymap).bindPopup("<b>Sangole</b><br />Area:  km2<br>Population: 28,116<br>villages: 103 <br> Avg. Rainfall:  mm<br><br><form action='villages.php' method='post'><input type='hidden'  value='Sangole' name='tehsil'><input type='hidden'  value='17.4341' name='lat'><input type='hidden'  value='75.1954' name='lon'><input type='submit' class='page-scroll btn btn-xl' style='border-radius: 3px;font-size: 10px; padding: 4px 6px ; '  value='Explore Sangole'></form>").openPopup();
	L.marker([17.561797, 76.074580]).addTo(mymap).bindPopup("<b>South Solapur</b><br />Area: 1,195.3 km2<br>Population:210,774<br>villages: 90<br> Avg. Rainfall: 617.33 mm <br><br><form action='villages.php' method='post'><input type='hidden'  value='17.561797' name='lat'><input type='hidden'  value='76.074580' name='lon'><input type='hidden'  value='SS' name='tehsil'><input type='submit' class='page-scroll btn btn-xl' style='border-radius: 3px;font-size: 10px; padding: 4px 6px ; '  value='Explore South Solapur'></form>").openPopup();
	L.marker([17.763138, 75.889277]).addTo(mymap).bindPopup("<b>North Solapur</b><br />Area: 711 km2<br>Population:960,803<br>villages: 54<br> Avg. Rainfall: 617.33 mm<br><br><form action='villages.php' method='post'><input type='hidden'  value='17.763138' name='lat'><input type='hidden'  value='75.889277' name='lon'><input type='hidden'  value='SN' name='tehsil'><input type='submit' class='page-scroll btn btn-xl' style='border-radius: 3px;font-size: 10px; padding: 4px 6px ; '  value='Explore North Solapur'></form>").openPopup();
	
	

var popup = L.popup();

	function onMapClick(e) {
		popup
			.setLatLng(e.latlng)
			.setContent("You clicked the map at " + e.latlng.toString())
			.openOn(mymap);
	}

	mymap.on('click', onMapClick);

</script>




    
    <!-- About Section -->
    <section id="about">
        <div class="container">
            <div class="row">
                <center>
                    <h3> - Our Mission - </h3>
                    <h6> To advice the farmers to cultivate the best crops so as to maximize the profits.</h6>
                    <br>
                    <h3> - About - </h3>
                    <p> Our goal is to help the farmers and Agricultural Industry in terms of Crop Produce, Crop Value as per Climatic Conditions, Market Value and User Inputs. Our system is a community based GIS System which maps the mupltiple parameters and provides an optimised result that benefits and advices the farmer and the buyers.</p>
            </div>
        </div>
    </section>

    <!-- Team Section -->
    <section id="team" class="bg-light-gray">
        <div class="container">
            <div class="row">
                <div class="col-lg-12 text-center">
                    <h2 class="section-heading">Our Amazing Team</h2>
                </div>
            </div>
            <div class="row">
               <div class="col-sm-4">
                    <div class="team-member">
                        <img src="pooja.jpg" class="img-responsive img-circle" alt="">
                        <h4>Pooja Bhatkar</h4>
                        <ul class="list-inline social-buttons">
                        
                            <li><a href="#"><i class="fa fa-facebook"></i></a>
                            </li>
                            <li><a href="#"><i class="fa fa-linkedin"></i></a>
                            </li>
                        </ul>
                    </div>
                </div>
                <div class="col-sm-4">
                    <div class="team-member">
                        <img src="Salome.jpeg" class="img-responsive img-circle" alt="">
                        <h4>Salome D'souza</h4>
                        <ul class="list-inline social-buttons">
                        
                            <li><a href="#"><i class="fa fa-facebook"></i></a>
                            </li>
                            <li><a href="#"><i class="fa fa-linkedin"></i></a>
                            </li>
                        </ul>
                    </div>
                </div>
                <div class="col-sm-4">
                    <div class="team-member">
                        <img src="nabeel.jpeg" class="img-responsive img-circle" alt="">
                        <h4>Nabeel Syed</h4>
                        <ul class="list-inline social-buttons">
                        
                            <li><a href="#"><i class="fa fa-facebook"></i></a>
                            </li>
                            <li><a href="#"><i class="fa fa-linkedin"></i></a>
                            </li>
                        </ul>
                    </div>
                </div>
                <div class="col-sm-4">
                    <div class="team-member">
                        <img src="harshad.png" class="img-responsive img-circle" alt="">
                        <h4>Harshad Vatsa</h4>
                        <ul class="list-inline social-buttons">
                            
                            <li><a href="#"><i class="fa fa-facebook"></i></a>
                            </li>
                            <li><a href="#"><i class="fa fa-linkedin"></i></a>
                            </li>
                        </ul>
                    </div>
                </div>
            </div>
            <div class="row">
                <div class="col-lg-8 col-lg-offset-2 text-center">
                    <p class="large text-muted"></p>
                </div>
            </div>
        </div>
    </section>

   
    <!-- Contact Section -->
    

    <footer>
        <div class="container">
            <div class="row">
                <div class="col-md-4">
                    <span class="copyright">Copyright &copy; Your Website 2016</span>
                </div>
                <div class="col-md-4">
                    <ul class="list-inline social-buttons">
                        <li><a href="#"><i class="fa fa-twitter"></i></a>
                        </li>
                        <li><a href="#"><i class="fa fa-facebook"></i></a>
                        </li>
                        <li><a href="#"><i class="fa fa-linkedin"></i></a>
                        </li>
                    </ul>
                </div>
                <div class="col-md-4">
                    <ul class="list-inline quicklinks">
                        <li><a href="#">Privacy Policy</a>
                        </li>
                        <li><a href="#">Terms of Use</a>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
    </footer>

    <!-- Portfolio Modals -->
    <!-- Use the modals below to showcase details about your portfolio projects! -->

    <!-- Portfolio Modal 1 -->
    <div class="portfolio-modal modal fade" id="portfolioModal1" tabindex="-1" role="dialog" aria-hidden="true">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="close-modal" data-dismiss="modal">
                    <div class="lr">
                        <div class="rl">
                        </div>
                    </div>
                </div>
                <div class="container">
                    <div class="row">
                        <div class="col-lg-8 col-lg-offset-2">
                            <div class="modal-body">
                                <!-- Project Details Go Here -->
                                <h2>Project Name</h2>
                                <p class="item-intro text-muted">Lorem ipsum dolor sit amet consectetur.</p>
                                <img class="img-responsive img-centered" src="img/portfolio/roundicons-free.jpg" alt="">
                                <p>Use this area to describe your project. Lorem ipsum dolor sit amet, consectetur adipisicing elit. Est blanditiis dolorem culpa incidunt minus dignissimos deserunt repellat aperiam quasi sunt officia expedita beatae cupiditate, maiores repudiandae, nostrum, reiciendis facere nemo!</p>
                                <p>
                                    <strong>Want these icons in this portfolio item sample?</strong>You can download 60 of them for free, courtesy of <a href="https://getdpd.com/cart/hoplink/18076?referrer=bvbo4kax5k8ogc">RoundIcons.com</a>, or you can purchase the 1500 icon set <a href="https://getdpd.com/cart/hoplink/18076?referrer=bvbo4kax5k8ogc">here</a>.</p>
                                <ul class="list-inline">
                                    <li>Date: July 2014</li>
                                    <li>Client: Round Icons</li>
                                    <li>Category: Graphic Design</li>
                                </ul>
                                <button type="button" class="btn btn-primary" data-dismiss="modal"><i class="fa fa-times"></i> Close Project</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Portfolio Modal 2 -->
    <div class="portfolio-modal modal fade" id="portfolioModal2" tabindex="-1" role="dialog" aria-hidden="true">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="close-modal" data-dismiss="modal">
                    <div class="lr">
                        <div class="rl">
                        </div>
                    </div>
                </div>
                <div class="container">
                    <div class="row">
                        <div class="col-lg-8 col-lg-offset-2">
                            <div class="modal-body">
                                <h2>Project Heading</h2>
                                <p class="item-intro text-muted">Lorem ipsum dolor sit amet consectetur.</p>
                                <img class="img-responsive img-centered" src="img/portfolio/startup-framework-preview.jpg" alt="">
                                <p><a href="http://designmodo.com/startup/?u=787">Startup Framework</a> is a website builder for professionals. Startup Framework contains components and complex blocks (PSD+HTML Bootstrap themes and templates) which can easily be integrated into almost any design. All of these components are made in the same style, and can easily be integrated into projects, allowing you to create hundreds of solutions for your future projects.</p>
                                <p>You can preview Startup Framework <a href="http://designmodo.com/startup/?u=787">here</a>.</p>
                                <button type="button" class="btn btn-primary" data-dismiss="modal"><i class="fa fa-times"></i> Close Project</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Portfolio Modal 3 -->
    <div class="portfolio-modal modal fade" id="portfolioModal3" tabindex="-1" role="dialog" aria-hidden="true">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="close-modal" data-dismiss="modal">
                    <div class="lr">
                        <div class="rl">
                        </div>
                    </div>
                </div>
                <div class="container">
                    <div class="row">
                        <div class="col-lg-8 col-lg-offset-2">
                            <div class="modal-body">
                                <!-- Project Details Go Here -->
                                <h2>Project Name</h2>
                                <p class="item-intro text-muted">Lorem ipsum dolor sit amet consectetur.</p>
                                <img class="img-responsive img-centered" src="img/portfolio/treehouse-preview.jpg" alt="">
                                <p>Treehouse is a free PSD web template built by <a href="https://www.behance.net/MathavanJaya">Mathavan Jaya</a>. This is bright and spacious design perfect for people or startup companies looking to showcase their apps or other projects.</p>
                                <p>You can download the PSD template in this portfolio sample item at <a href="http://freebiesxpress.com/gallery/treehouse-free-psd-web-template/">FreebiesXpress.com</a>.</p>
                                <button type="button" class="btn btn-primary" data-dismiss="modal"><i class="fa fa-times"></i> Close Project</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Portfolio Modal 4 -->
    <div class="portfolio-modal modal fade" id="portfolioModal4" tabindex="-1" role="dialog" aria-hidden="true">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="close-modal" data-dismiss="modal">
                    <div class="lr">
                        <div class="rl">
                        </div>
                    </div>
                </div>
                <div class="container">
                    <div class="row">
                        <div class="col-lg-8 col-lg-offset-2">
                            <div class="modal-body">
                                <!-- Project Details Go Here -->
                                <h2>Project Name</h2>
                                <p class="item-intro text-muted">Lorem ipsum dolor sit amet consectetur.</p>
                                <img class="img-responsive img-centered" src="img/portfolio/golden-preview.jpg" alt="">
                                <p>Start Bootstrap's Agency theme is based on Golden, a free PSD website template built by <a href="https://www.behance.net/MathavanJaya">Mathavan Jaya</a>. Golden is a modern and clean one page web template that was made exclusively for Best PSD Freebies. This template has a great portfolio, timeline, and meet your team sections that can be easily modified to fit your needs.</p>
                                <p>You can download the PSD template in this portfolio sample item at <a href="http://freebiesxpress.com/gallery/golden-free-one-page-web-template/">FreebiesXpress.com</a>.</p>
                                <button type="button" class="btn btn-primary" data-dismiss="modal"><i class="fa fa-times"></i> Close Project</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Portfolio Modal 5 -->
    <div class="portfolio-modal modal fade" id="portfolioModal5" tabindex="-1" role="dialog" aria-hidden="true">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="close-modal" data-dismiss="modal">
                    <div class="lr">
                        <div class="rl">
                        </div>
                    </div>
                </div>
                <div class="container">
                    <div class="row">
                        <div class="col-lg-8 col-lg-offset-2">
                            <div class="modal-body">
                                <!-- Project Details Go Here -->
                                <h2>Project Name</h2>
                                <p class="item-intro text-muted">Lorem ipsum dolor sit amet consectetur.</p>
                                <img class="img-responsive img-centered" src="img/portfolio/escape-preview.jpg" alt="">
                                <p>Escape is a free PSD web template built by <a href="https://www.behance.net/MathavanJaya">Mathavan Jaya</a>. Escape is a one page web template that was designed with agencies in mind. This template is ideal for those looking for a simple one page solution to describe your business and offer your services.</p>
                                <p>You can download the PSD template in this portfolio sample item at <a href="http://freebiesxpress.com/gallery/escape-one-page-psd-web-template/">FreebiesXpress.com</a>.</p>
                                <button type="button" class="btn btn-primary" data-dismiss="modal"><i class="fa fa-times"></i> Close Project</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- Portfolio Modal 6 -->
    <div class="portfolio-modal modal fade" id="portfolioModal6" tabindex="-1" role="dialog" aria-hidden="true">
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="close-modal" data-dismiss="modal">
                    <div class="lr">
                        <div class="rl">
                        </div>
                    </div>
                </div>
                <div class="container">
                    <div class="row">
                        <div class="col-lg-8 col-lg-offset-2">
                            <div class="modal-body">
                                <!-- Project Details Go Here -->
                                <h2>Project Name</h2>
                                <p class="item-intro text-muted">Lorem ipsum dolor sit amet consectetur.</p>
                                <img class="img-responsive img-centered" src="img/portfolio/dreams-preview.jpg" alt="">
                                <p>Dreams is a free PSD web template built by <a href="https://www.behance.net/MathavanJaya">Mathavan Jaya</a>. Dreams is a modern one page web template designed for almost any purpose. Its a beautiful template thats designed with the Bootstrap framework in mind.</p>
                                <p>You can download the PSD template in this portfolio sample item at <a href="http://freebiesxpress.com/gallery/dreams-free-one-page-web-template/">FreebiesXpress.com</a>.</p>
                                <button type="button" class="btn btn-primary" data-dismiss="modal"><i class="fa fa-times"></i> Close Project</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <!-- jQuery -->
    <script src="vendor/jquery/jquery.min.js"></script>

    <!-- Bootstrap Core JavaScript -->
    <script src="vendor/bootstrap/js/bootstrap.min.js"></script>

    <!-- Plugin JavaScript -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/jquery-easing/1.3/jquery.easing.min.js" integrity="sha384-mE6eXfrb8jxl0rzJDBRanYqgBxtJ6Unn4/1F7q4xRRyIw7Vdg9jP4ycT7x1iVsgb" crossorigin="anonymous"></script>

    <!-- Contact Form JavaScript -->
    <script src="js/jqBootstrapValidation.js"></script>
    <script src="js/contact_me.js"></script>

    <!-- Theme JavaScript -->
    <script src="js/agency.min.js"></script>
    <script>
    	var x = document.getElementById("demo");

    	function getLocation() {
    	  if (navigator.geolocation) {
    	    navigator.geolocation.getCurrentPosition(showPosition);
    	  } else { 
    	    x.innerHTML = "Geolocation is not supported by this browser.";
    	  }
    	}

    	function showPosition(position) {
    	  x.innerHTML = "Latitude: " + position.coords.latitude + 
    	  "<br>Longitude: " + position.coords.longitude;
    	  document.getElementById('lat').value = position.coords.latitude;
    	  document.getElementById('lon').value = position.coords.longitude;
    	}
</script>

</body>

</html>

