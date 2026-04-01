import cadquery as cq

sketch1 = cq.Sketch().rect(2, 2)
sketch2 = cq.Sketch().circle(1)
result = (
cq.Workplane("XY")
.box(20, 20, 2)
.faces(">Z")
.workplane()
.placeSketch(sketch1, sketch2)
cq.exporters.export(result, 'GT.stl')