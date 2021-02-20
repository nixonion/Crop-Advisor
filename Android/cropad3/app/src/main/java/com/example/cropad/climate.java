package com.example.cropad;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.TextView;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;

public class climate extends AppCompatActivity implements AdapterView.OnItemSelectedListener {

    String spin_village,spin_district,clim;
    TextView textView,line1,line2,line3,line4,line5,line6,line7;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_climate);
        Spinner spinner_d=findViewById(R.id.dictrict);
        ArrayAdapter<CharSequence> adapter1=ArrayAdapter.createFromResource(this, R.array.district,android.R.layout.simple_spinner_item);
        adapter1.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner_d.setAdapter(adapter1);
        spinner_d.setOnItemSelectedListener(this);

        textView = (TextView) findViewById(R.id.tvout);
        line1 = (TextView) findViewById(R.id.line1);
        line2 = (TextView) findViewById(R.id.line2);
        line3 = (TextView) findViewById(R.id.line3);
        line4 = (TextView) findViewById(R.id.line4);
        line5 = (TextView) findViewById(R.id.line5);
        line6 = (TextView) findViewById(R.id.line6);
        line7 = (TextView) findViewById(R.id.line7);




        final RequestQueue queue = Volley.newRequestQueue(this);

        Button send_request = findViewById(R.id.request);

        send_request.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {

                //textView.setText(spin_village+","+spin_district);
                String url ="http://ec2-18-223-164-111.us-east-2.compute.amazonaws.com/put123/scrape.php?district="+spin_district+"&village="+spin_village;
                StringRequest stringRequest = new StringRequest(Request.Method.GET, url,
                        new Response.Listener<String>() {
                            @Override
                            public void onResponse(String response) {
                                // Display the first 500 characters of the response string.
                                clim=response.toString();
                                if(!clim.equals('0')) {
                                    String[] wet = clim.split(",", 7);
                                    wet[1] = wet[1].replaceAll("\\s+","");
                                    wet[2] = wet[2].replaceAll("\\s+","");
                                    line1.setText(wet[0]);
                                    line2.setText(wet[1]);
                                    line3.setText(wet[2]);
                                    line4.setText(wet[3]);
                                    line5.setText(wet[4]);
                                    line6.setText(wet[5]);
                                    line7.setText(wet[6]);

                                }
                                //textView.setText(clim);
                            }
                        }, new Response.ErrorListener() {
                    @Override
                    public void onErrorResponse(VolleyError error) {
                        clim="That didn't work!\n Connection Problem";
                        textView.setText(clim);
                    }
                });

                queue.add(stringRequest);
            }
        });

    }

    @Override
    public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {

        switch (adapterView.getId())
        {
            case R.id.dictrict:
                spin_district= adapterView.getItemAtPosition(i).toString();
                //textView.setText(spin_district);
                Spinner spinner_v=findViewById(R.id.villages);
                ArrayAdapter<CharSequence> adapter2=null;
                switch (spin_district)
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
            case R.id.villages:
                spin_village= adapterView.getItemAtPosition(i).toString();
                //textView.setText(spin_village+","+spin_district);
                break;
        }

    }

    @Override
    public void onNothingSelected(AdapterView<?> adapterView) {

    }
}
