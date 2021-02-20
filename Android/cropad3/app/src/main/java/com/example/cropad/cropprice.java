package com.example.cropad;

import android.app.DatePickerDialog;
import android.content.Context;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.Gravity;
import android.view.LayoutInflater;
import android.view.View;
import android.widget.Button;
import android.widget.DatePicker;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.PopupWindow;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;

import java.util.Calendar;

public class cropprice extends AppCompatActivity implements RadioGroup.OnCheckedChangeListener {

    String futcropp,futuredate1,resprice;
    private int mYear, mMonth, mDay;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_cropprice);

        final TextView outsmall= (TextView) findViewById(R.id.outline1);
        final TextView outbig= (TextView) findViewById(R.id.outline2);

        RadioGroup radio;
        radio = findViewById(R.id.fucrop);
        radio.setOnCheckedChangeListener(this);

        final RequestQueue queue = Volley.newRequestQueue(this);

        final EditText futuredate= (EditText) findViewById(R.id.futuredate);
        futuredate.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                // TODO Auto-generated method stub
                final Calendar c = Calendar.getInstance();
                mYear = c.get(Calendar.YEAR);
                mMonth = c.get(Calendar.MONTH);
                mDay = c.get(Calendar.DAY_OF_MONTH);


                DatePickerDialog datePickerDialog = new DatePickerDialog(cropprice.this,
                        new DatePickerDialog.OnDateSetListener() {

                            @Override
                            public void onDateSet(DatePicker view, int year,
                                                  int monthOfYear, int dayOfMonth) {

                                futuredate1=year+"-"+(monthOfYear + 1)+"-"+dayOfMonth;
                                futuredate.setText(dayOfMonth + "-" + (monthOfYear + 1) + "-" + year);

                            }
                        }, mYear, mMonth, mDay);
                datePickerDialog.show();
            }
        });

        Button getprice = findViewById(R.id.getresult);
        getprice.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //out1.setText(water1);Toast.makeText(getApplicationContext(), "Not Working", Toast.LENGTH_LONG).show();
                String url ="http://ec2-18-223-164-111.us-east-2.compute.amazonaws.com/reg.php?crop="+futcropp+"&date="+futuredate1;


                StringRequest stringRequest = new StringRequest(Request.Method.GET, url,
                        new Response.Listener<String>() {

                            public void onResponse(String response) {
                                resprice=response.toString();
                                outsmall.setText("Price of "+futcropp+" on "+futuredate1+" is:");
                                outbig.setText(resprice);
                            }

                        }, new Response.ErrorListener() {
                    @Override
                    public void onErrorResponse(VolleyError error) {
                        Toast.makeText(getApplicationContext(), "Not Working", Toast.LENGTH_LONG).show();
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
            case R.id.jowar:
                futcropp="Jowar";
                break;
            case R.id.maize:
                futcropp="Maize";
                break;
            case R.id.bajra:
                futcropp="Bajra";
                break;
            case R.id.wheat:
                futcropp="Wheat";
                break;
        }

    }
}
