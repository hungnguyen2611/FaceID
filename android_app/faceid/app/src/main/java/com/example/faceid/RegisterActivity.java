package com.example.faceid;


import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.text.Editable;
import android.text.TextWatcher;
import android.util.Base64;
import android.util.Log;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageSwitcher;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.android.volley.Cache;
import com.android.volley.Network;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.toolbox.BasicNetwork;
import com.android.volley.toolbox.DiskBasedCache;
import com.android.volley.toolbox.HurlStack;
import com.android.volley.toolbox.JsonObjectRequest;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;


public class RegisterActivity extends AppCompatActivity {
    Button previous, next, upload_img_btn, backbtn, done;
    ImageSwitcher imageView;
    int PICK_IMAGE_MULTIPLE = 3;
    String user_name;
    ArrayList<Uri> mArrayUri = new ArrayList<>();
    ArrayList<Bitmap> mArrayBmp = new ArrayList<>();
    ArrayList<String> mArrayEncodedString = new ArrayList<>();
    String requested_url = "http://03e5-1-52-129-151.ngrok.io/register";
    RequestQueue queue;
    int position = 0;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_register);

        backbtn = findViewById(R.id.button_back_to_main);
        backbtn.setOnClickListener(v -> startActivity(new Intent(RegisterActivity.this, MainActivity.class)));
        EditText name_input = findViewById(R.id.name_input);
        final TextView textView = findViewById(R.id.textviewid);
        name_input.addTextChangedListener(new TextWatcher() {
            @Override
            public void beforeTextChanged(CharSequence charSequence, int i, int i1, int i2) {

            }

            @Override
            public void onTextChanged(CharSequence charSequence, int i, int i1, int i2) {

            }

            @Override
            public void afterTextChanged(Editable editable) {
                textView.setText(name_input.getText().toString());
            }
        });



        imageView = findViewById(R.id.image);


        // showing all images in imageswitcher
        imageView.setFactory(() -> new ImageView(getApplicationContext()));

        next = findViewById(R.id.next);
        // click here to select next image
        next.setOnClickListener(v -> {
            if (position < mArrayUri.size() - 1) {
                // increase the position by 1
                position++;
                imageView.setImageURI(mArrayUri.get(position));
            } else {
                Toast.makeText(RegisterActivity.this, "Last Image Already Shown", Toast.LENGTH_SHORT).show();
            }
        });
        previous = findViewById(R.id.previous);
        // click here to view previous image
        previous.setOnClickListener(v -> {
            if (position > 0) {
                // decrease the position by 1
                position--;
                imageView.setImageURI(mArrayUri.get(position));
            }
        });

        imageView = findViewById(R.id.image);

        upload_img_btn = findViewById(R.id.button_upload);
        upload_img_btn.setOnClickListener(v -> {

            // initialising intent
            Intent intent = new Intent();

            // setting type to select to be image
            intent.setType("image/*");

            // allowing multiple image to be selected
            intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
            intent.setAction(Intent.ACTION_GET_CONTENT);
            startActivityForResult(Intent.createChooser(intent, "Select Picture"), PICK_IMAGE_MULTIPLE);
        });

        done = findViewById(R.id.button_done);
        done.setOnClickListener(v -> {
            user_name = name_input.getText().toString();
            if (user_name.equals("") || mArrayBmp.size() != 3) {
                if (user_name.equals(""))
                    Toast.makeText(this, "User name isn't allowed to be empty string", Toast.LENGTH_LONG).show();
                if (mArrayBmp.size() != 3)
                    Toast.makeText(this, "Upload 3 images, please :)", Toast.LENGTH_LONG).show();
            }
            else{
                for (int i = 0; i < mArrayBmp.size(); i++){
                    try (ByteArrayOutputStream bos = new ByteArrayOutputStream()) {
                        //create a file to write bitmap data
                        mArrayBmp.get(i).compress(Bitmap.CompressFormat.JPEG, 100, bos);
                        byte[] img_data = bos.toByteArray();
                        String image = Base64.encodeToString(img_data, Base64.DEFAULT);
                        mArrayEncodedString.add(image);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                sendImagetoURL(user_name, mArrayEncodedString, requested_url);
                mArrayBmp.clear();
                mArrayEncodedString.clear();
                mArrayUri.clear();
                user_name="";
                startActivity(new Intent(RegisterActivity.this, MainActivity.class));
            }
        });


    }

    private Bitmap downscaleBitmap(Bitmap originalBitmap, int expected_size) {
        float width_ratio = (float)originalBitmap.getWidth() / expected_size;
        float height_ratio = (float) originalBitmap.getHeight() / expected_size;
        float final_ratio = width_ratio > height_ratio? width_ratio : height_ratio;

        return Bitmap.createScaledBitmap(originalBitmap, (int)(originalBitmap.getWidth()/final_ratio), (int)(originalBitmap.getHeight()/final_ratio), false);
    }

    private void sendImagetoURL(String user_name, ArrayList<String> mArrayEncodedString, String requested_url) {
        // Instantiate the cache
        Cache cache = new DiskBasedCache(getCacheDir(), 1024 * 1024); // 1MB cap

        // Set up the network to use HttpURLConnection as the HTTP client.
        Network network = new BasicNetwork(new HurlStack());
        // Instantiate the RequestQueue with the cache and network.
        queue = new RequestQueue(cache, network);
        queue.start();

        HashMap<String, Object> data_map = new HashMap<>();
        data_map.put("user_name", user_name);
        data_map.put("first_img", mArrayEncodedString.get(0));
        data_map.put("sec_img", mArrayEncodedString.get(1));
        data_map.put("third_img", mArrayEncodedString.get(2));
        JsonObjectRequest jsonObjectRequest = new JsonObjectRequest
                (Request.Method.POST, requested_url, new JSONObject(data_map), response -> {
                    boolean res;
                    String name;
                    try {
                        res = response.getBoolean("Existed");
                        name = response.getString("Name");
                        if (!res)
                            Toast.makeText(getApplicationContext(), String.format("Successfully added new user %s to database", name), Toast.LENGTH_LONG).show();
                        else Toast.makeText(getApplicationContext(), String.format("Found %s in database, attempting to add more images...", name), Toast.LENGTH_LONG).show();
                    }
                    catch (JSONException e){
                        e.printStackTrace();
                    }

                }, error -> {
                    // TODO: Handle error
                    Toast.makeText(getApplicationContext(), "something went wrong", Toast.LENGTH_LONG).show();
                    Log.e("Volly Error", error.toString());
                });
        queue.add(jsonObjectRequest);
    }


    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        // When an Image is picked
        if (requestCode == PICK_IMAGE_MULTIPLE && resultCode == RESULT_OK && null != data) {
            // Get the Image from data
            if (data.getClipData() != null) {
                int cout = data.getClipData().getItemCount();
                for (int i = 0; i < cout; i++) {
                    // adding imageuri in array
                    Uri imageurl = data.getClipData().getItemAt(i).getUri();
                    mArrayUri.add(imageurl);
                    try {
                        mArrayBmp.add(MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageurl));
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
                // setting 1st selected image into image switcher
            } else {
                Uri imageurl = data.getData();
                mArrayUri.add(imageurl);
                try {
                    Bitmap bmp = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageurl);
                    bmp = downscaleBitmap(bmp, 400);
                    mArrayBmp.add(bmp);
                } catch (IOException e) {
                    e.printStackTrace();
                }
            }
            imageView.setImageURI(mArrayUri.get(0));
            position = 0;
        } else {
            // show this if no image is selected
            Toast.makeText(this, "You haven't picked Image", Toast.LENGTH_LONG).show();
        }
    }
}