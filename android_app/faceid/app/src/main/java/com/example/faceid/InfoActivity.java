package com.example.faceid;

import androidx.appcompat.app.AppCompatActivity;

import android.annotation.SuppressLint;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import java.util.Base64;

public class InfoActivity extends AppCompatActivity {
    
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_info);
        Intent intent = getIntent();
        String image_encode_str = intent.getStringExtra("user_image");
        byte[] decodedBytes = Base64.getDecoder().decode(image_encode_str);
        Bitmap bitmap = BitmapFactory.decodeByteArray(decodedBytes , 0, decodedBytes.length);
        ImageView mImageView;
        mImageView = (ImageView) findViewById(R.id.imageViewId);
        mImageView.setImageBitmap(bitmap);


        Button back_btn = findViewById(R.id.button_back_to_main);
        back_btn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                startActivity(new Intent(InfoActivity.this, MainActivity.class));
            }
        });

        TextView textView = (TextView)findViewById(R.id.hello_user_id);
        textView.setText(String.format("Name:%s", intent.getStringExtra("user_name"))); //set text for text view
    }



}