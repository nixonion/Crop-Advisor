package com.example.cropad;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.location.Location;
import android.support.constraint.ConstraintLayout;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.PopupWindow;
import android.widget.RadioButton;
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.TextView;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;
import com.google.android.gms.location.FusedLocationProviderClient;
import com.google.android.gms.location.LocationServices;
import com.google.android.gms.tasks.OnSuccessListener;

public class Input extends AppCompatActivity implements RadioGroup.OnCheckedChangeListener, AdapterView.OnItemSelectedListener {

    String cbut,spinbut,area1,water1,marketp,spin_district1,spin_village1,ph1,temp1;
    private FusedLocationProviderClient mFusedLocationClient;
    ConstraintLayout con;
    PopupWindow popupWindow;
    TextView result;
    Double lat, lon;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_input);
        RadioGroup radioGroup;
        RadioButton radioButton;
        radioGroup = findViewById(R.id.rad);
        con = findViewById(R.id.cropselect);

        final TextView out1= (TextView) findViewById(R.id.out);
        Button send = findViewById(R.id.send);

        final RequestQueue queue = Volley.newRequestQueue(this);

        final EditText area= (EditText) findViewById(R.id.area);
        final EditText water= (EditText) findViewById(R.id.water);
        final EditText ph= (EditText) findViewById(R.id.ph);
        final EditText temp= (EditText) findViewById(R.id.temp);



        Spinner spinner=findViewById(R.id.crop);
        ArrayAdapter<CharSequence> adapter=ArrayAdapter.createFromResource(this, R.array.district,android.R.layout.simple_spinner_item);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);
        spinner.setOnItemSelectedListener(this);

        mFusedLocationClient = LocationServices.getFusedLocationProviderClient(this);

        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) != PackageManager.PERMISSION_GRANTED && ActivityCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) != PackageManager.PERMISSION_GRANTED) {

            /*LayoutInflater layoutInflater = (LayoutInflater) Input.this.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
            View customView = layoutInflater.inflate(R.layout.nolocation,null);
            //popupWindow = new PopupWindow(customView, LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT);
            popupWindow = new PopupWindow(customView, LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT);
            popupWindow.showAtLocation(con, Gravity.CENTER, 0, 0);*/



            // TODO: Consider calling
            //    ActivityCompat#requestPermissions
            // here to request the missing permissions, and then overriding
            //   public void onRequestPermissionsResult(int requestCode, String[] permissions,
            //                                          int[] grantResults)
            // to handle the case where the user grants the permission. See the documentation
            // for ActivityCompat#requestPermissions for more details.

            //return;
        }
        mFusedLocationClient.getLastLocation()
                .addOnSuccessListener(this, new OnSuccessListener<Location>() {
                    @Override
                    public void onSuccess(Location location) {
                        // Got last known location. In some rare situations this can be null.
                        if (location != null) {
                            // Logic to handle location object
                            lat = location.getLatitude();
                            lon = location.getLongitude();
                            //name=name+lon.toString()+lat.toString();


                        }
                    }
                });



        radioGroup.setOnCheckedChangeListener(this);
        send.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                LayoutInflater layoutInflater = (LayoutInflater) Input.this.getSystemService(Context.LAYOUT_INFLATER_SERVICE);
                View customView = layoutInflater.inflate(R.layout.pop,null);
                //popupWindow = new PopupWindow(customView, LinearLayout.LayoutParams.WRAP_CONTENT, LinearLayout.LayoutParams.WRAP_CONTENT);
                popupWindow = new PopupWindow(customView, LinearLayout.LayoutParams.FILL_PARENT, LinearLayout.LayoutParams.FILL_PARENT);
                final Button close = (Button) customView.findViewById(R.id.closeb);
                //display the popup window
                result= (TextView) customView.findViewById(R.id.res);
                area1=area.getText().toString();
                water1=water.getText().toString();
                ph1=ph.getText().toString();
                temp1=temp.getText().toString();
                //out1.setText(water1);
                String url ="http://ec2-18-223-164-111.us-east-2.compute.amazonaws.com/test/test.php?taluka="+spin_district1+"&village="+spin_village1+"&area="+area1+"&water="+water1+"&soil="+cbut+"&lat="+lat.toString()+"&lon="+lon.toString()+"&ph="+ph1+"&temp="+temp1;
                StringRequest stringRequest = new StringRequest(Request.Method.GET, url,
                        new Response.Listener<String>() {
                            @Override
                            public void onResponse(String response) {
                                // Display the first 500 characters of the response string.
                                marketp=response.toString();
                                //out1.setText(marketp);
                                result.setText(marketp);
                                popupWindow.showAtLocation(con, Gravity.CENTER, 0, 0);
                                close.setOnClickListener(new View.OnClickListener() {
                                    @Override
                                    public void onClick(View v) {
                                        popupWindow.dismiss();
                                    }
                                });

                            }
                        }, new Response.ErrorListener() {
                    @Override
                    public void onErrorResponse(VolleyError error) {
                        marketp="That didn't work!\n Connection Problem";

                        //popupWindow.showAtLocation(con, Gravity.CENTER, 0, 0);
                        out1.setText(marketp);
                    }
                });

                queue.add(stringRequest);
            }
        });



    }



    @Override
    public void onCheckedChanged(RadioGroup radioGroup, int i) {
        switch (i)
        {
            case R.id.deep:
                cbut="Deep_Black";
                break;
            case R.id.medium:
                cbut="Medium_Deep";
                break;
            case R.id.shallow:
                cbut="Shallow";
                break;
        }

    }

    @Override
    public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
        //spinbut= adapterView.getItemAtPosition(i).toString();

        switch (adapterView.getId())
        {
            case R.id.crop:
                spin_district1= adapterView.getItemAtPosition(i).toString();
                //textView.setText(spin_district);
                Spinner spinner_v=findViewById(R.id.vl);
                ArrayAdapter<CharSequence> adapter2=null;
                switch (spin_district1)
                {

                    case "Akkalkot":
                        adapter2=ArrayAdapter.createFromResource(this, R.array.akkalkot,android.R.layout.simple_spinner_item);
                        break;
                    case "Barshi":
                        adapter2=ArrayAdapter.createFromResource(this, R.array.barshi,android.R.layout.simple_spinner_item);
                        break;
                    case "Karmala":
                        adapter2=ArrayAdapter.createFromResource(this, R.array.karmala,android.R.layout.simple_spinner_item);
                        break;
                    case "Madha":
                        adapter2=ArrayAdapter.createFromResource(this, R.array.madha,android.R.layout.simple_spinner_item);
                        break;
                    case "Malshiras":
                        adapter2=ArrayAdapter.createFromResource(this, R.array.malshiras,android.R.layout.simple_spinner_item);
                        break;
                    case "Mangalvedhe":
                        adapter2=ArrayAdapter.createFromResource(this, R.array.mangalvedhe,android.R.layout.simple_spinner_item);
                        break;
                    case "Mohol":
                        adapter2=ArrayAdapter.createFromResource(this, R.array.mohol,android.R.layout.simple_spinner_item);
                        break;
                    case "Pandharpur":
                        adapter2=ArrayAdapter.createFromResource(this, R.array.pandharpur,android.R.layout.simple_spinner_item);
                        break;
                    case "Sangole":
                        adapter2=ArrayAdapter.createFromResource(this, R.array.sangole,android.R.layout.simple_spinner_item);
                        break;
                    case "Solapur-North":
                        adapter2=ArrayAdapter.createFromResource(this, R.array.solapur_n,android.R.layout.simple_spinner_item);
                        break;
                    case "South-Solapur":
                        adapter2=ArrayAdapter.createFromResource(this, R.array.s_solapur,android.R.layout.simple_spinner_item);
                        break;
                }
                adapter2.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
                spinner_v.setAdapter(adapter2);
                spinner_v.setOnItemSelectedListener(this);

                break;
            case R.id.vl:
                spin_village1= adapterView.getItemAtPosition(i).toString();
                //textView.setText(spin_village+","+spin_district);
                break;
        }

    }

    @Override
    public void onNothingSelected(AdapterView<?> adapterView) {

    }
}
