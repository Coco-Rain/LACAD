import cadquery as cq

sketch1 = cq.Sketch().circle(1)
sketch2 = cq.Sketch().circle(0.5)
result =(
cq.Workplane("XY")
.placeSketch(sketch1, sketch2.moved(z=5))
.loft(combine= True)
)
cq.exporters.export(result, 'GT.stl')