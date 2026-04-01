import cadquery as cq

frame = (
cq.Sketch()
.arc((0,0), 80, 20, -20)
.segment((100, 0),(80, 0))
.arc((0,0), 100, 0, 20)
.close()
.assemble(tag="face")
)
result = cq.Workplane("XY").placeSketch(frame).extrude(340)
cq.exporters.export(result, 'GT.stl')