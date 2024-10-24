///Select the vector data to be clipped/选择需要裁剪的矢量数据
var aoi = ee.FeatureCollection("users/akidou0923/AreaofGuiyang");
//(You need to manually upload the research area shp file to GEE's personal space
//需要手动上传研究区域shp文件到GEE的个人空间)

//Load vector borders to facilitate selecting sample points within the boundary/加载矢量边框，以便于在边界内选取样本点
var empty = ee.Image().toByte();
var outline = empty.paint({
 featureCollection:aoi, // The administrative boundary is named fc/行政边界命名为fc
 color:0, //color transparent/颜色透明
 width:3 //boundary width/边界宽度
});
Map.addLayer(outline, {palette: "ff0000"}, "outline");

//cloud removal function去云函数 
function cloudmaskL7sr(image) {
  var qa = image.select('QA_PIXEL');
  var cloudMask = qa.bitwiseAnd(1 << 3).eq(0).and(qa.bitwiseAnd(1 << 4).eq(0));
  return image.updateMask(cloudMask);
}
//Landsat and sentinel have different cloud removal functions. Please refer to the official documents or literature
//landsat和sentinel有不同的去云函数，请参考官方文档或文献资料

// Applies scaling factors.
function applyScaleFactors(image) {
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermalBand = image.select('ST_B6').multiply(0.00341802).add(149.0);
  return image.addBands(opticalBands, null, true)
              .addBands(thermalBand, null, true);
}
//select raster datasets/选择栅格数据集 
var dataset = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
  .filterDate('2009-01-01', '2010-12-31')
  .filterBounds(aoi)
  .map(cloudmaskL7sr);
  
dataset = dataset.map(applyScaleFactors);

var composite = dataset.median().clip(aoi);
//Display image collection results for easy sample selection/显示图像收集结果方便选取样本
Map.addLayer(composite, {bands: ['SR_B3', 'SR_B2', 'SR_B1'], min: 0, max: 0.3}, 'Color (321)');
Map.addLayer(composite, {bands: ['SR_B4', 'SR_B3', 'SR_B2'], min: 0, max: 0.3}, 'Color (432)');
Map.addLayer(composite, {bands: ['SR_B7', 'SR_B5', 'SR_B4'], min: 0, max: 0.3}, 'Color (754)');
print("Landsa7 Image Collection",dataset);

// Gets a timestamp for each image in the image collection/获取图像集合中每个图像的时间戳
//This step is to observe which images are involved in the synthesis/这一步是为了观察有那些图像参与了合成
var dates = dataset.aggregate_array('system:time_start');
dates = dates.map(function(d) {
  return ee.Date(d).format('YYYY-MM-dd');
});
// Print the date of all images/打印所有图像的日期
print('Image Dates in Collection:', dates);
var dem = ee.Image("NASA/NASADEM_HGT/001")
// Construct Classfication Dataset
// RS Index Cacluate(NDVI\NDWI\EVI\BSI)
//需要根据不同数据集进行参数名称修改，如landsat5和landsat8的波段名称不相同
//Parameter names need to be modified according to different data sets. For example, the band names of landsat5 and landsat8 are different.
var add_RS_index = function(img){
  var ndvi = img.normalizedDifference(['SR_B4', 'SR_B3']).rename('NDVI').copyProperties(img,['system:time_start']);
  var ndwi = img.normalizedDifference(['SR_B2', 'SR_B4']).rename('NDWI').copyProperties(img,['system:time_start']);
  var evi = img.expression('2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', 
  {
    'NIR': img.select('SR_B4'),
    'RED': img.select('SR_B3'),
    'BLUE': img.select('SR_B1')
  }).rename('EVI').copyProperties(img,['system:time_start']);
  var bsi = img.expression('((RED + SWIR1) - (NIR + BLUE)) / ((RED + SWIR1) + (NIR + BLUE)) ', 
  {
    'RED': img.select('SR_B3'), 
    'BLUE': img.select('SR_B1'),
    'NIR': img.select('SR_B4'),
    'SWIR1': img.select('SR_B5'),

  }).rename('BSI').copyProperties(img,['system:time_start']);


  var ibi = img.expression('(2 * SWIR1 / (SWIR1 + NIR) - (NIR / (NIR + RED) + GREEN / (GREEN + SWIR1))) / (2 * SWIR1 / (SWIR1 + NIR) + (NIR / (NIR + RED) + GREEN / (GREEN + SWIR1)))', {
    'SWIR1': img.select('SR_B5'),
    'NIR': img.select('SR_B4'),
    'RED': img.select('SR_B3'),
    'GREEN': img.select('SR_B2')
  }).rename('IBI').copyProperties(img,['system:time_start']);
  return img.addBands([ndvi, ndwi, evi, bsi, ibi]);
};
var dataset = dataset.map(add_RS_index); 
var bands = ['SR_B1','SR_B2','SR_B3','SR_B4','SR_B5','SR_B7','NDVI','NDWI','BSI'];
var imgcol_median = dataset.select(bands).median();
var aoi_dem = dem.select('elevation').clip(aoi).rename('DEM');
var construct_img = imgcol_median.addBands(aoi_dem).clip(aoi);

//classification sample/分类样本
var train_points = Cropland.merge(Grassland).merge(Buildup).merge(Forest).merge(Water);//Modify based on classification samples/根据分类样本进行修改
var train_data= construct_img.sampleRegions({
  collection: train_points,
  properties: ['landcover'],//This way needs to be the same as the attribute name of the selected point/此处需要与选取点的属性名称相同
  scale: 30
});

//accuracy evaluation/精度评价
var withRandom = train_data.randomColumn('random');//Random arrangement of sample points/样本点随机的排列
var split = 0.7; 
var trainingPartition = withRandom.filter(ee.Filter.lt('random', split));//Screening 70% of the samples as training samples/筛选70%的样本作为训练样本
var testingPartition = withRandom.filter(ee.Filter.gte('random', split));//Screening 30% of the samples as test samples/筛选30%的样本作为测试样本

//Classification method Select random forest/分类方法选择随机森林
var rf = ee.Classifier.smileRandomForest({
numberOfTrees: 20,
bagFraction: 0.8
}).train({
features: trainingPartition,
classProperty: 'landcover',//This way needs to be the same as the attribute name of the selected point/此处需要与选取点的属性名称相同
inputProperties: bands
});

//Perform random forest classification on sentinel data/对哨兵数据进行随机森林分类
var img_classfication = construct_img.classify(rf); 

//Use test sample classification to determine the data set and function to perform functional operations
//运用测试样本分类，确定要进行函数运算的数据集以及函数
var test = testingPartition.classify(rf);

//Calculate the confusion matrix/计算混淆矩阵
var confusionMatrix = test.errorMatrix('landcover', 'classification');
print('confusionMatrix',confusionMatrix);//Confusion matrix displayed on panel/面板上显示混淆矩阵
print('overall accuracy', confusionMatrix.accuracy());//Overall accuracy displayed on the panel/面板上显示总体精度
print('kappa accuracy', confusionMatrix.kappa());//kappa value displayed on the panel/面板上显示kappa值
Map.centerObject(aoi)
Map.addLayer(aoi);
Map.addLayer(img_classfication.clip(aoi), {min: 1, max: 5, palette: ['orange', 'blue', 'green','yellow','red']});//Color is set based on the number of categories, and the value of max also needs to be changed/根据分类数量设定颜色，max的值也需要更改
var class1=img_classfication.clip(aoi)

//Export classification map/导出分类图
Export.image.toDrive({  
       image: class1,  
       description: 'rfclass2009-2010',  
       fileNamePrefix: 'rf2009-2010',  //file naming/文件命名
       folder: "class",  //the saved folder/保存的文件夹
       scale: 30,  //resolution/分辨率
       region: aoi,  //study area/研究区
       maxPixels: 1e13,  //Maximum image element, just default/最大像元素，默认就好
       crs: "EPSG:4326"  //set the projection/设置投影
   });