package com.example.cropad;

import android.content.Intent;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.Rect;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.TableLayout;
import android.widget.TableRow;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonArrayRequest;
import com.android.volley.toolbox.StringRequest;
import com.android.volley.toolbox.Volley;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

public class rates extends AppCompatActivity implements AdapterView.OnItemSelectedListener {
    String text,marketp;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_rates);

        final RequestQueue queue = Volley.newRequestQueue(this);

        final TableLayout table = findViewById(R.id.table);
        TableRow row = new TableRow(this);

        String[] head= {"Commodity","Min","Max","Modal","Date"};


        for (int j = 0; j < 5;j++) {

            TextView cell = new TextView(this);
            cell.setBackgroundColor(Color.WHITE);
            cell.setGravity(Gravity.CENTER);
            cell.setText(head[j]);
            cell.setPadding(40, 4, 20, 4);
            row.addView(cell);
        }
        table.addView(row);
        /*for (int i = 0; i < 10; i++) {

            TableRow row = new TableRow(this);
            row.setBackgroundColor(000000);


            TableLayout.LayoutParams tableRowParams=
                    new TableLayout.LayoutParams
                            (TableLayout.LayoutParams.FILL_PARENT,TableLayout.LayoutParams.WRAP_CONTENT);

            tableRowParams.setMargins(1,1,1,1);
            tableRowParams.weight=1;
            row.setLayoutParams(tableRowParams);


            for (int j = 0; j < 10; j++) {

                TextView cell = new TextView(this);
                cell.setBackgroundColor(Color.WHITE);
                cell.setGravity(Gravity.CENTER);


                cell.setText(i + ", " + j);
                cell.setPadding(10, 1, 10, 1);
                row.addView(cell);

            }

            table.addView(row);
        }*/



        Spinner spinner=findViewById(R.id.spinner);
        ArrayAdapter<CharSequence> adapter=ArrayAdapter.createFromResource(this, R.array.market,android.R.layout.simple_spinner_item);
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner.setAdapter(adapter);
        spinner.setOnItemSelectedListener(this);

        final TextView test= (TextView) findViewById(R.id.test);
        final Button market = findViewById(R.id.button);
        market.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                //test.setText(text);

                table.removeAllViews();
                String[] head= {"Commodity","Min","Max","Modal","Date"};
                TableRow row = new TableRow(rates.this);


                for (int j = 0; j < 5;j++) {

                    TextView cell = new TextView(rates.this);
                    cell.setBackgroundColor(Color.WHITE);
                    cell.setGravity(Gravity.CENTER);
                    cell.setText(head[j]);
                    cell.setPadding(40, 4, 20, 4);
                    row.addView(cell);
                }
                table.addView(row);

                //hereeeeeeeeeeeeeeeee
                String url ="http://ec2-18-223-164-111.us-east-2.compute.amazonaws.com/rates.php?name="+text;
                JsonArrayRequest req= new JsonArrayRequest(url, new Response.Listener<JSONArray>() {
                    @Override
                    public void onResponse(JSONArray response) {
                        try {
                            int n=response.length();

                            for (int i = 0; i < response.length(); i++) {
                                JSONObject object = (JSONObject) response.getJSONObject(i);
                                String[] js= new String[5];
                                js[0] = object.getString("commodity");
                                js[1] = object.getString("min");
                                js[2] = object.getString("max");
                                js[3] = object.getString("modal");
                                js[4] = object.getString("date");

                                TableRow row = new TableRow(rates.this);

                                for (int j = 0; j < 5;j++) {

                                    TextView cell = new TextView(rates.this);
                                    cell.setBackgroundColor(Color.WHITE);
                                    cell.setGravity(Gravity.CENTER);
                                    cell.setText(js[j]);
                                    cell.setPadding(40, 4, 20, 4);
                                    row.addView(cell);
                                }
                                table.addView(row);



                            }

                        }
                        catch (JSONException e)
                        {
                            e.printStackTrace();
                            Toast.makeText(getApplicationContext(),"Exception: " + e.getMessage(), Toast.LENGTH_LONG).show();
                        }

                    }
                }, new Response.ErrorListener() {
                    @Override
                    public void onErrorResponse(VolleyError error) {
                        Toast.makeText(getApplicationContext(),"No Data to Plot Graph ", Toast.LENGTH_LONG).show();

                    }
                });
                queue.add(req);
                //hereeeeeeeeeeeeeeee

                //getrate(text);
            }
        });



    }



    @Override
    public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {
        text= adapterView.getItemAtPosition(i).toString();
    }

    @Override
    public void onNothingSelected(AdapterView<?> adapterView) {

    }
}
