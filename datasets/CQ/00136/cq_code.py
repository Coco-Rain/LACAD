import cadquery as cq

sk = cq.Sketch().rect(20, 50).vertices().fillet(5)
result = (
cq.Workplane("XY")
.placeSketch(sk)
.extrude(55)
.translate((0, 0, -22.5))
)
cq.exporters.export(result, 'GT.stl')