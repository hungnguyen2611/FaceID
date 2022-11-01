// Copyright 2021 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.example.faceid;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.opengl.GLES20;
import android.util.Base64;
import android.util.Log;
import android.widget.Toast;

import com.android.volley.Cache;
import com.android.volley.Network;
import com.android.volley.Request;
import com.android.volley.RequestQueue;
import com.android.volley.Response;
import com.android.volley.VolleyError;
import com.android.volley.toolbox.BasicNetwork;
import com.android.volley.toolbox.DiskBasedCache;
import com.android.volley.toolbox.HurlStack;
import com.android.volley.toolbox.JsonObjectRequest;
import com.google.mediapipe.solutioncore.ResultGlRenderer;
import com.google.mediapipe.solutions.facedetection.FaceDetectionResult;
import com.google.mediapipe.solutions.facedetection.FaceKeypoint;
import com.google.mediapipe.formats.proto.DetectionProto.Detection;

import org.json.JSONException;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.HashMap;
import java.util.Calendar;
import java.util.Objects;

/** A custom implementation of {@link ResultGlRenderer} to render {@link FaceDetectionResult}. */
public class FaceDetectionResultGlRenderer implements ResultGlRenderer<FaceDetectionResult> {
    private static final String TAG = "FaceDetectionResultGlRenderer";

    private static final float[] KEYPOINT_COLOR = new float[] {1f, 0f, 0f, 1f};
    private static final float KEYPOINT_SIZE = 16f;
    private static final float[] BBOX_COLOR = new float[] {0f, 1f, 0f, 1f};
    private static final int BBOX_THICKNESS = 8;
    private static final String VERTEX_SHADER =
            "uniform mat4 uProjectionMatrix;\n"
                    + "uniform float uPointSize;\n"
                    + "attribute vec4 vPosition;\n"
                    + "void main() {\n"
                    + "  gl_Position = uProjectionMatrix * vPosition;\n"
                    + "  gl_PointSize = uPointSize;"
                    + "}";
    private static final String FRAGMENT_SHADER =
            "precision mediump float;\n"
                    + "uniform vec4 uColor;\n"
                    + "void main() {\n"
                    + "  gl_FragColor = uColor;\n"
                    + "}";
    private int program;
    private int positionHandle;
    private int pointSizeHandle;
    private int projectionMatrixHandle;
    private int colorHandle;
    private String requested_url = "http://03e5-1-52-129-151.ngrok.io/predict";

    private RequestQueue queue;
    private Context ctx;
    private int cam_width, cam_height;
    private long counter  = System.currentTimeMillis();
    private boolean detected = false;
    private int loadShader(int type, String shaderCode) {
        int shader = GLES20.glCreateShader(type);
        GLES20.glShaderSource(shader, shaderCode);
        GLES20.glCompileShader(shader);
        return shader;
    }

    public FaceDetectionResultGlRenderer(Context context, int width, int height){
        this.ctx = context;
        this.cam_width = width;
        this.cam_height = height;
    }
    @Override
    public void setupRendering() {
        program = GLES20.glCreateProgram();
        int vertexShader = loadShader(GLES20.GL_VERTEX_SHADER, VERTEX_SHADER);
        int fragmentShader = loadShader(GLES20.GL_FRAGMENT_SHADER, FRAGMENT_SHADER);
        GLES20.glAttachShader(program, vertexShader);
        GLES20.glAttachShader(program, fragmentShader);
        GLES20.glLinkProgram(program);
        positionHandle = GLES20.glGetAttribLocation(program, "vPosition");
        pointSizeHandle = GLES20.glGetUniformLocation(program, "uPointSize");
        projectionMatrixHandle = GLES20.glGetUniformLocation(program, "uProjectionMatrix");
        colorHandle = GLES20.glGetUniformLocation(program, "uColor");
    }

    @Override
    public void renderResult(FaceDetectionResult result, float[] projectionMatrix) {
        long t = System.currentTimeMillis();
        long elapsed = t - counter;
        if (result == null) {
            return;
        }
        GLES20.glUseProgram(program);
        GLES20.glUniformMatrix4fv(projectionMatrixHandle, 1, false, projectionMatrix, 0);
        GLES20.glUniform1f(pointSizeHandle, KEYPOINT_SIZE);
        int numDetectedFaces = result.multiFaceDetections().size();
        for (int i = 0; i < numDetectedFaces; ++i) {

            Detection detection_result = result.multiFaceDetections().get(i);
            float score = result.multiFaceDetections().get(i).getScore(0);
            if (score >= 0.9) {
                if (elapsed >= 500) {
                    counter = System.currentTimeMillis();
                    float xmin = detection_result.getLocationData().getRelativeBoundingBox().getXmin();
                    float ymin = detection_result.getLocationData().getRelativeBoundingBox().getYmin();
                    float width = detection_result.getLocationData().getRelativeBoundingBox().getWidth();
                    float height = detection_result.getLocationData().getRelativeBoundingBox().getHeight();
                    float[] points = new float[FaceKeypoint.NUM_KEY_POINTS * 2];
                    for (int idx = 0; idx < FaceKeypoint.NUM_KEY_POINTS; ++idx) {
                        points[2 * idx] = detection_result.getLocationData().getRelativeKeypoints(idx).getX();
                        points[2 * idx + 1] = detection_result.getLocationData().getRelativeKeypoints(idx).getY();
                    }
                    try {
                        cutFrame(xmin, ymin, width, height, this.cam_width, this.cam_height, points, requested_url);
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }
            drawDetection(detection_result);
        }
    }

    /**
     * Deletes the shader program.
     *
     * <p>This is only necessary if one wants to release the program while keeping the context around.
     */
    public void release() {
        GLES20.glDeleteProgram(program);
    }

    private void drawDetection(Detection detection) {
        if (!detection.hasLocationData()) {
            return;
        }
        // Draw keypoints.
        float[] points = new float[FaceKeypoint.NUM_KEY_POINTS * 2];
        for (int i = 0; i < FaceKeypoint.NUM_KEY_POINTS; ++i) {
            points[2 * i] = detection.getLocationData().getRelativeKeypoints(i).getX();
            points[2 * i + 1] = detection.getLocationData().getRelativeKeypoints(i).getY();
        }
        GLES20.glUniform4fv(colorHandle, 1, KEYPOINT_COLOR, 0);
        FloatBuffer vertexBuffer =
                ByteBuffer.allocateDirect(points.length * 4)
                        .order(ByteOrder.nativeOrder())
                        .asFloatBuffer()
                        .put(points);
        vertexBuffer.position(0);
        GLES20.glEnableVertexAttribArray(positionHandle);
        GLES20.glVertexAttribPointer(positionHandle, 2, GLES20.GL_FLOAT, false, 0, vertexBuffer);
        GLES20.glDrawArrays(GLES20.GL_POINTS, 0, FaceKeypoint.NUM_KEY_POINTS);
        if (!detection.getLocationData().hasRelativeBoundingBox()) {
            return;
        }
        // Draw bounding box.
        float left = detection.getLocationData().getRelativeBoundingBox().getXmin();
        float top = detection.getLocationData().getRelativeBoundingBox().getYmin();
        float right = left + detection.getLocationData().getRelativeBoundingBox().getWidth();
        float bottom = top + detection.getLocationData().getRelativeBoundingBox().getHeight();
        drawLine(top, left, top, right);
        drawLine(bottom, left, bottom, right);
        drawLine(top, left, bottom, left);
        drawLine(top, right, bottom, right);
    }

    private void drawLine(float y1, float x1, float y2, float x2) {
        GLES20.glUniform4fv(colorHandle, 1, BBOX_COLOR, 0);
        GLES20.glLineWidth(BBOX_THICKNESS);
        float[] vertex = {x1, y1, x2, y2};
        FloatBuffer vertexBuffer =
                ByteBuffer.allocateDirect(vertex.length * 4)
                        .order(ByteOrder.nativeOrder())
                        .asFloatBuffer()
                        .put(vertex);
        vertexBuffer.position(0);
        GLES20.glEnableVertexAttribArray(positionHandle);
        GLES20.glVertexAttribPointer(positionHandle, 2, GLES20.GL_FLOAT, false, 0, vertexBuffer);
        GLES20.glDrawArrays(GLES20.GL_LINES, 0, 2);
    }


    @SuppressLint("LongLogTag")
    public void cutFrame(float x, float y, float width, float height, int mWidth, int mHeight, float[] points, String requested_url) throws IOException {
        float[] detection = {x, y, width, height};
        // transform detection bbox
        int[] transformedDetection = {(int)(detection[0]*mWidth), (int)(detection[1]*mHeight),
                (int)(detection[2]*mWidth), (int)(detection[3]*mHeight)};
        byte[] img_data = null;
        // transform keypoints to pixels
        int[] pixel_points = pixelKeypointHelper(points, mWidth, mHeight);
        // keypoints after cropped
        int[] transformedKeypoint = {
                pixel_points[0] - transformedDetection[0],
                pixel_points[1] - transformedDetection[1],
                pixel_points[2] - transformedDetection[0],
                pixel_points[3] - transformedDetection[1]
        };

        Bitmap bmp = savePixelsHelper( 0, 0, mWidth, mHeight);
        //Crop bitmap image based on detection
        bmp = Bitmap.createBitmap(bmp, transformedDetection[0], transformedDetection[1],
                transformedDetection[2], transformedDetection[3]);
        try (ByteArrayOutputStream bos = new ByteArrayOutputStream()) {
            //create a file to write bitmap data
            bmp.compress(Bitmap.CompressFormat.JPEG, 100, bos);
            img_data = bos.toByteArray();
            sendImagetoURL(img_data, transformedKeypoint, requested_url);
        }
    }

    private int[] pixelKeypointHelper(float[] points, int mWidth, int mHeight){
        int[] pixel_keypoint = new int[FaceKeypoint.NUM_KEY_POINTS*2];
        for (int i = 0; i < FaceKeypoint.NUM_KEY_POINTS; ++i) {
            pixel_keypoint[2 * i] = (int) (points[2 * i] * mWidth);
            pixel_keypoint[2 * i + 1] = (int) (points[2 * i + 1] * mHeight);
        }
        return pixel_keypoint;
    }

    private Bitmap savePixelsHelper(int x, int y, int mWidth, int mHeight){

        ByteBuffer buffer = ByteBuffer.allocateDirect(mWidth * mHeight * 4);
        GLES20.glReadPixels(x, y, mWidth, mHeight, GLES20.GL_RGBA,
                GLES20.GL_UNSIGNED_BYTE, buffer);
        //Create bitmap preview resolution
        Bitmap bitmap = Bitmap.createBitmap(mWidth, mHeight, Bitmap.Config.ARGB_8888);
        //Set buffer to bitmap
        bitmap.copyPixelsFromBuffer(buffer);
        //Scale to stream resolution
        //Flip vertical
        return createFlippedBitmap(bitmap, false, true);
    }
    public static Bitmap createFlippedBitmap(Bitmap source, boolean xFlip, boolean yFlip) {
        Matrix matrix = new Matrix();
        matrix.postScale(xFlip ? -1 : 1, yFlip ? -1 : 1, source.getWidth() / 2f, source.getHeight() / 2f);
        return Bitmap.createBitmap(source, 0, 0, source.getWidth(), source.getHeight(), matrix, true);
    }


    private void sendImagetoURL(byte[] data, int[] keypoints, String requested_url){

        String url = requested_url;
        // Instantiate the cache
        Cache cache = new DiskBasedCache(ctx.getCacheDir(), 1024 * 1024); // 1MB cap

        // Set up the network to use HttpURLConnection as the HTTP client.
        Network network = new BasicNetwork(new HurlStack());
        // Instantiate the RequestQueue with the cache and network.
        queue = new RequestQueue(cache, network);
        queue.start();
        String image = Base64.encodeToString(data, Base64.DEFAULT);
        String file_name = String.valueOf(Calendar.getInstance().getTimeInMillis());

        HashMap<String, Object> data_map = new HashMap<>();
        data_map.put("file_name", file_name);
        data_map.put("image", image);
        data_map.put("keypoints_x1", keypoints[0]);
        data_map.put("keypoints_y1", keypoints[1]);
        data_map.put("keypoints_x2", keypoints[2]);
        data_map.put("keypoints_y2", keypoints[3]);

        JsonObjectRequest jsonObjectRequest = new JsonObjectRequest
                (Request.Method.POST, url, new JSONObject(data_map), new Response.Listener<JSONObject>(){
                    @SuppressLint("LongLogTag")
                    @Override
                    public void onResponse(JSONObject response) {
                        String res = null;
                        if (detected){
                            queue.cancelAll(new RequestQueue.RequestFilter() {
                                @Override
                                public boolean apply(Request<?> request) {
                                    return true;
                                }
                            });
                            return;
                        }
                        try {
                            res = response.getString("Result");
                        }
                        catch (JSONException e){
                            e.printStackTrace();
                        }
                        if (!Objects.equals(res, "Unknown")){
                            detected = true;
                            Toast.makeText(ctx.getApplicationContext(), "Hello " + res + ". Welcome back :D", Toast.LENGTH_LONG).show();
                            Intent myIntent = new Intent(ctx, InfoActivity.class);
                            try {
                                myIntent.putExtra("user_image", response.getString("user_image"));
                                myIntent.putExtra("user_name", res);
                            } catch (JSONException e) {
                                e.printStackTrace();
                            }
                            ctx.startActivity(myIntent);
                        }
                        Log.i(TAG, res);

                    }
                }, new Response.ErrorListener() {
                    @SuppressLint("LongLogTag")
                    @Override
                    public void onErrorResponse(VolleyError error) {
                        // TODO: Handle error
                        Toast.makeText(ctx.getApplicationContext(), "something went wrong", Toast.LENGTH_LONG).show();
                        Log.e("Volly Error", error.toString());
                    }
                });
        jsonObjectRequest.setTag(detected);
        queue.add(jsonObjectRequest);
    }

}



