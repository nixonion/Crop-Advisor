package com.example.cropad;

import android.app.DatePickerDialog;
import android.graphics.Color;
import android.graphics.Point;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.AdapterView;
import android.widget.ArrayAdapter;
import android.widget.Button;
import android.widget.DatePicker;
import android.widget.EditText;
import android.widget.RadioGroup;
import android.widget.Spinner;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.JsonArrayRequest;
import com.android.volley.toolbox.Volley;
import com.jjoe64.graphview.GraphView;
import com.jjoe64.graphview.LegendRenderer;
import com.jjoe64.graphview.helper.DateAsXAxisLabelFormatter;
import com.jjoe64.graphview.helper.StaticLabelsFormatter;
import com.jjoe64.graphview.series.DataPoint;
import com.jjoe64.graphview.series.DataPointInterface;
import com.jjoe64.graphview.series.LineGraphSeries;
import com.jjoe64.graphview.series.OnDataPointTapListener;
import com.jjoe64.graphview.series.Series;

import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

import java.util.Calendar;
import java.util.Date;
import java.util.Random;

public class analysis extends AppCompatActivity implements RadioGroup.OnCheckedChangeListener,AdapterView.OnItemSelectedListener{

    String crop,location,fromdate,todate;
    int min,max,modal,selop;
    private int mYear, mMonth, mDay,mYear1, mMonth1, mDay1;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_analysis);

        Button anal = findViewById(R.id.analyse);
        final GraphView graph = (GraphView) findViewById(R.id.graph);
        final RequestQueue queue = Volley.newRequestQueue(this);

        final TextView txtDate=  findViewById(R.id.textView8);

        final EditText to= (EditText) findViewById(R.id.to);
        final EditText from= (EditText) findViewById(R.id.from);


        to.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                // TODO Auto-generated method stub
                final Calendar c = Calendar.getInstance();
                mYear = c.get(Calendar.YEAR);
                mMonth = c.get(Calendar.MONTH);
                mDay = c.get(Calendar.DAY_OF_MONTH);


                DatePickerDialog datePickerDialog = new DatePickerDialog(analysis.this,
                        new DatePickerDialog.OnDateSetListener() {

                            @Override
                            public void onDateSet(DatePicker view, int year,
                                                  int monthOfYear, int dayOfMonth) {

                                todate=year+"-"+(monthOfYear + 1)+"-"+dayOfMonth;
                                to.setText(dayOfMonth + "-" + (monthOfYear + 1) + "-" + year);

                            }
                        }, mYear, mMonth, mDay);
                datePickerDialog.show();
            }
        });
        from.setOnClickListener(new View.OnClickListener() {

            @Override
            public void onClick(View v) {
                // TODO Auto-generated method stub
                final Calendar c = Calendar.getInstance();
                mYear = c.get(Calendar.YEAR);
                mMonth = c.get(Calendar.MONTH);
                mDay = c.get(Calendar.DAY_OF_MONTH);


                DatePickerDialog datePickerDialog = new DatePickerDialog(analysis.this,
                        new DatePickerDialog.OnDateSetListener() {

                            @Override
                            public void onDateSet(DatePicker view, int year,
                                                  int monthOfYear, int dayOfMonth) {

                                fromdate=year+"-"+(monthOfYear + 1)+"-"+dayOfMonth;

                                from.setText(dayOfMonth + "-" + (monthOfYear + 1) + "-" + year);

                            }
                        }, mYear, mMonth, mDay);
                datePickerDialog.show();

            }
        });


        RadioGroup radio;
        radio = findViewById(R.id.option);
        radio.setOnCheckedChangeListener(this);



        Spinner spinner123=findViewById(R.id.locate);
        ArrayAdapter<CharSequence> adapter123=ArrayAdapter.createFromResource(this, R.array.market,android.R.layout.simple_spinner_item);
        adapter123.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner123.setAdapter(adapter123);
        spinner123.setOnItemSelectedListener(this);

        Spinner spinner456=findViewById(R.id.crop12);
        ArrayAdapter<CharSequence> adapter456=ArrayAdapter.createFromResource(this, R.array.crop,android.R.layout.simple_spinner_item);
        adapter456.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item);
        spinner456.setAdapter(adapter456);
        spinner456.setOnItemSelectedListener(this);

        anal.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                graph.removeAllSeries();
                String url;
                if (selop==1)
                {
                    Toast.makeText(getApplicationContext(),"Generating Graph ", Toast.LENGTH_LONG).show();
                    url ="http://ec2-18-223-164-111.us-east-2.compute.amazonaws.com/graph1.php?crop="+crop+"&location="+location+"&from="+fromdate+"&to="+todate;
                }
                else
                {
                    Toast.makeText(getApplicationContext(),"Generating Graph ", Toast.LENGTH_LONG).show();
                    url ="http://ec2-18-223-164-111.us-east-2.compute.amazonaws.com/graph.php?crop="+crop+"&location="+location;

                }
                //JsonArrayRequest arrayRequest = new JsonArrayRequest(Request.Method.GET,url, null, null);
                JsonArrayRequest req= new JsonArrayRequest(url, new Response.Listener<JSONArray>() {
                    @Override
                    public void onResponse(JSONArray response) {
                        try {
                            int n=response.length();
                            DataPoint[] dp=new DataPoint[n];
                            DataPoint[] low=new DataPoint[n];
                            DataPoint[] up=new DataPoint[n];
                            String[] xco= new String[n];
                            Date[] xc01=new Date[n];
                            if(n>=1) {


                                //Toast.makeText(getApplicationContext(),"Done "+response.length(), Toast.LENGTH_LONG).show();
                                for (int i = 0; i < response.length(); i++) {
                                    JSONObject object = (JSONObject) response.getJSONObject(i);
                                    String commodity = object.getString("commodity");
                                    Calendar cal = Calendar.getInstance();
                                    cal.set(1995, 5, 25);
                                    String date = object.getString("date");
                                    String[] da = date.split("-", 3);
                                    xco[i] = da[2] + "." + da[1];
                                    cal.set(Integer.parseInt(da[0]), Integer.parseInt(da[1]), Integer.parseInt(da[2]));
                                    xc01[i] = cal.getTime();
                                    min = Integer.parseInt(object.getString("min"));
                                    max = Integer.parseInt(object.getString("max"));
                                    modal = Integer.parseInt(object.getString("modal"));
                                    DataPoint v = new DataPoint(i, modal);
                                    DataPoint v1 = new DataPoint(i, min);
                                    DataPoint v2 = new DataPoint(i, max);
                                    dp[i] = v;
                                    low[i] = v1;
                                    up[i] = v2;

                                    Toast.makeText(getApplicationContext(), "Graph is Generated ", Toast.LENGTH_LONG).show();
                                }
                            /*
                            int count=30;
                            Random mRand = new Random();
                            DataPoint[] values = new DataPoint[count];
                            for (int i=0; i<count; i++) {
                                double x = i;
                                double f = mRand.nextDouble()*0.15+0.3;
                                double y = Math.sin(i*f+2) + mRand.nextDouble()*0.3;
                                DataPoint v = new DataPoint(x, y);
                                values[i] = v;
                            }*/
                                StaticLabelsFormatter staticLabelsFormatter = new StaticLabelsFormatter(graph);
                                staticLabelsFormatter.setHorizontalLabels(xco);
                                graph.getGridLabelRenderer().setLabelFormatter(staticLabelsFormatter);

                                //graph.getGridLabelRenderer().setLabelFormatter(new DateAsXAxisLabelFormatter(analysis.this));
                                //
                                //graph.getGridLabelRenderer().setNumHorizontalLabels(n);
                                graph.getViewport().setScalable(true);
                                //graph.getGridLabelRenderer().setHumanRounding(false);


                                LineGraphSeries<DataPoint> series1 = new LineGraphSeries<>(dp);
                                LineGraphSeries<DataPoint> series2 = new LineGraphSeries<>(low);
                                LineGraphSeries<DataPoint> series3 = new LineGraphSeries<>(up);
                                /*
                            LineGraphSeries<DataPoint> series = new LineGraphSeries<>(new DataPoint[] {
                                    new DataPoint(0, 1),
                                    new DataPoint(1, 5),
                                    new DataPoint(2, 3),
                                    new DataPoint(3, 2),
                                    new DataPoint(4, 6)
                            });*/
                                graph.getViewport().setXAxisBoundsManual(true);
                                graph.getViewport().setMinX(series1.getLowestValueX());
                                graph.getViewport().setMaxX(series1.getHighestValueX());
                                graph.addSeries(series1);
                                graph.addSeries(series2);
                                graph.addSeries(series3);
                                series1.setDrawDataPoints(true);
                                series1.setDataPointsRadius(10);
                                series1.setTitle("Mean");
                                series2.setTitle("Min");
                                series3.setTitle("Max");
                                series2.setColor(Color.GREEN);
                                series3.setColor(Color.RED);
                                graph.getLegendRenderer().setVisible(true);
                                graph.getLegendRenderer().setAlign(LegendRenderer.LegendAlign.TOP);
                                series1.setOnDataPointTapListener(new OnDataPointTapListener() {
                                    @Override
                                    public void onTap(Series series, DataPointInterface dataPoint) {
                                        Toast.makeText(getApplicationContext(), "Point : " + dataPoint, Toast.LENGTH_LONG).show();
                                    }
                                });
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
                        //marketp="That didn't work!\n Connection Problem";
                        //out1.setText(marketp);
                    }
                });
                queue.add(req);





            }
        });

    }

    @Override
    public void onItemSelected(AdapterView<?> adapterView, View view, int i, long l) {

        switch (adapterView.getId()) {

            case R.id.locate:
                location= adapterView.getItemAtPosition(i).toString();

                break;
            case R.id.crop12:

                crop= adapterView.getItemAtPosition(i).toString();
                break;

        }

    }

    @Override
    public void onNothingSelected(AdapterView<?> adapterView) {

    }

    @Override
    public void onCheckedChanged(RadioGroup radioGroup, int i) {

        switch (i)
        {
            case R.id.yes:
                selop=1;
                break;
            case R.id.no:
                selop=0;
                break;
         }

    }
}
