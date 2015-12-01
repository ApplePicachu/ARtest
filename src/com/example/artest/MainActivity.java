package com.example.artest;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

import org.artoolkit.ar.base.ARActivity;
import org.artoolkit.ar.base.rendering.ARRenderer;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.widget.FrameLayout;
import android.widget.ImageView;

public class MainActivity extends ARActivity {
	private String TAG = "AR";
	private ImageView iv;
	private ImageView imageViewDraw;
	private Mat imageMat;
	private Mat imageSubMat;
	private Bitmap imageBmp;
	private Bitmap imageDraw;
	private boolean markerVisible[] = { false, false, false };
	private CascadeClassifier faceDetector;
	private final float faceMinSizeRelative = 0.2f;
	private float faceMinSizeAbsolute;
	private Rect lastFacePos;
	private Bitmap glassBitmap;
	private Bitmap hatBitmap;
	private Bitmap maskBitmap;
	private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {

		@Override
		public void onManagerConnected(int status) {
			switch (status) {
			case LoaderCallbackInterface.SUCCESS: {
				Log.i(TAG, "OpenCV loaded successfully");
				imageMat = new Mat(240 / 3, 320, CvType.CV_8UC3);
				imageBmp = Bitmap.createBitmap(320/3, 240 / 3, Bitmap.Config.ARGB_8888);
				imageMat.setTo(new Scalar(0, 0, 0));
				lastFacePos = new Rect();
				try {
					// load cascade file from application resources
					InputStream is = getResources().openRawResource(R.raw.lbpcascade_frontalface);
					File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
					File mCascadeFile = new File(cascadeDir, "lbpcascade_frontalface.xml");
					FileOutputStream os = new FileOutputStream(mCascadeFile);

					byte[] buffer = new byte[4096];
					int bytesRead;
					while ((bytesRead = is.read(buffer)) != -1) {
						os.write(buffer, 0, bytesRead);
					}
					is.close();
					os.close();

					faceDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
					if (faceDetector.empty()) {
						Log.e(TAG, "Failed to load cascade classifier");
						faceDetector = null;
					} else
						Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

					cascadeDir.delete();

				} catch (IOException e) {
					e.printStackTrace();
					Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
				}
			}
				break;
			default:
				super.onManagerConnected(status);
				break;
			}
		}
	};

	@Override
	public void cameraPreviewStarted(int width, int height, int rate, int cameraIndex, boolean cameraIsFrontFacing) {
		Log.d(TAG, width + " " + height);
		super.cameraPreviewStarted(width, height, rate, cameraIndex, cameraIsFrontFacing);
		faceMinSizeAbsolute = (width < height ? width/3 : height/3) * faceMinSizeRelative;
	}

	@Override
	public void cameraPreviewFrame(byte[] frame) {
		imageMat.put(0, 0, frame);
		imageSubMat = imageMat.submat(0, imageMat.rows(), 0, imageMat.cols()/3);
		
		imageDraw.eraseColor(Color.TRANSPARENT);
		Canvas canvas = new Canvas(imageDraw);
		if(detectFace(imageSubMat)){
			if (markerVisible[0]) {
				drawGlasses(lastFacePos, canvas);
			}
			if (markerVisible[1]) {
				drawHat(lastFacePos, canvas);
			}
			if (markerVisible[2]) {
				drawMask(lastFacePos, canvas);
			}
		}
		imageViewDraw.setImageBitmap(imageDraw);
		Utils.matToBitmap(imageSubMat, imageBmp);
		iv.setImageBitmap(imageBmp);
		super.cameraPreviewFrame(frame);
	}

	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		setContentView(R.layout.activity_main);
		imageDraw = Bitmap.createBitmap(320, 240, Bitmap.Config.ARGB_8888);
		iv = (ImageView) findViewById(R.id.imageView1);
		imageViewDraw = (ImageView) findViewById(R.id.imageViewDraw);
		glassBitmap = BitmapFactory.decodeResource(getResources(), R.drawable.glass);
		hatBitmap = BitmapFactory.decodeResource(getResources(), R.drawable.hat);
		maskBitmap = BitmapFactory.decodeResource(getResources(), R.drawable.mask);
	}

	@Override
	public void onResume() {
		if (!OpenCVLoader.initDebug()) {
			Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
			OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
		} else {
			Log.d(TAG, "OpenCV library found inside package. Using it!");
			mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
		}
		super.onResume();
	}

	@Override
	public boolean onCreateOptionsMenu(Menu menu) {
		// Inflate the menu; this adds items to the action bar if it is present.
		getMenuInflater().inflate(R.menu.main, menu);
		return true;
	}

	@Override
	public boolean onOptionsItemSelected(MenuItem item) {
		// Handle action bar item clicks here. The action bar will
		// automatically handle clicks on the Home/Up button, so long
		// as you specify a parent activity in AndroidManifest.xml.
		int id = item.getItemId();
		if (id == R.id.action_settings) {
			return true;
		}
		return super.onOptionsItemSelected(item);
	}

	@Override
	protected ARRenderer supplyRenderer() {
		return new SimpleRenderer(markerVisible);
	}

	@Override
	protected FrameLayout supplyFrameLayout() {
		return (FrameLayout) findViewById(R.id.ARFrameLayout);
	}

	private void drawGlasses(Rect faceRect, Canvas canvas) {
		int drawL;
		int drawT;
		int drawR;
		int drawB;
		int moveDown = (int)((double)faceRect.height*0.2);
		drawL = faceRect.x;
		drawR = faceRect.x + faceRect.width;
		drawT = faceRect.y+moveDown;
		drawB = faceRect.y + (int)((double)faceRect.width/glassBitmap.getWidth()*glassBitmap.getHeight())+moveDown;
		drawL *= 3;
		drawR *= 3;
		drawT *= 3;
		drawB *= 3;
		android.graphics.Rect drawRect = new android.graphics.Rect(drawL, drawT, drawR, drawB);
		canvas.drawBitmap(glassBitmap, new android.graphics.Rect(0, 0, glassBitmap.getWidth(), glassBitmap.getHeight()), drawRect, null);
	}

	private void drawHat(Rect faceRect, Canvas canvas) {
		final double WIDTH_SCALE = 1.3;
		int drawL;
		int drawT;
		int drawR;
		int drawB;
		int moveUP = (int)(faceRect.height*0.6);
		double drawWidth = faceRect.width*WIDTH_SCALE;
		drawL = (int)(faceRect.x-(double)faceRect.width*(WIDTH_SCALE-1.0)/2.0);
		drawR = (int)(faceRect.x+drawWidth);
		drawT = faceRect.y-moveUP;
		drawB = (int)(faceRect.y+drawWidth/hatBitmap.getWidth()*hatBitmap.getHeight())-moveUP;
		drawL *= 3;
		drawR *= 3;
		drawT *= 3;
		drawB *= 3;
		android.graphics.Rect drawRect = new android.graphics.Rect(drawL, drawT, drawR, drawB);
		canvas.drawBitmap(hatBitmap, new android.graphics.Rect(0, 0, hatBitmap.getWidth(), hatBitmap.getHeight()), drawRect, null);
	}

	private void drawMask(Rect faceRect, Canvas canvas) {
		int drawL;
		int drawT;
		int drawR;
		int drawB;
		int moveDown = (int) ((double) faceRect.height * 0.6);
		drawL = faceRect.x;
		drawR = faceRect.x + faceRect.width;
		drawT = faceRect.y + moveDown;
		drawB = faceRect.y + (int) ((double) faceRect.width / maskBitmap.getWidth() * maskBitmap.getHeight()) + moveDown;
		drawL *= 3;
		drawR *= 3;
		drawT *= 3;
		drawB *= 3;
		android.graphics.Rect drawRect = new android.graphics.Rect(drawL, drawT, drawR, drawB);
		canvas.drawBitmap(maskBitmap, new android.graphics.Rect(0, 0, maskBitmap.getWidth(), maskBitmap.getHeight()), drawRect, null);
	}

	private boolean detectFace(Mat image) {
		MatOfRect face = new MatOfRect();
		faceDetector.detectMultiScale(image, face, 1.1, 2, 2, new Size(faceMinSizeAbsolute, faceMinSizeAbsolute), new Size());
		if (face.empty()) {
			lastFacePos = null;
			return false;
		}
		else {
			lastFacePos = face.toArray()[0].clone();
			Imgproc.rectangle(image, lastFacePos.tl(), lastFacePos.br(), new Scalar(255, 0, 0));
			return true;
		}
		
	}
}
