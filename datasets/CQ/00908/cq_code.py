import cadquery as cq

result = (
cq.Workplane("XY")
.center(-10, 0)
.vLine(3)
.threePointArc((10, 6),(20, 3))
.vLine(-3)
.mirrorX()
.extrude(30.0,True)
)
cq.exporters.export(result, 'GT.stl')