package com.example.cropad;

import android.content.Intent;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;


public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        Button price = findViewById(R.id.price);
        Button form = findViewById(R.id.form);
        Button etc = findViewById(R.id.etc);
        Button anal = findViewById(R.id.button2);
        Button future = findViewById(R.id.button3);
        final TextView out= (TextView) findViewById(R.id.yes);

        price.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startActivity(new Intent(MainActivity.this, rates.class));
            }
        });

        form.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startActivity(new Intent(MainActivity.this, Input.class));
            }
        });

        etc.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startActivity(new Intent(MainActivity.this, climate.class));
                //out.setText("This functionality is Not complete yet");

            }
        });
        anal.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startActivity(new Intent(MainActivity.this, analysis.class));
            }
        });

        future.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                startActivity(new Intent(MainActivity.this, cropprice.class));
            }
        });


    }
}
